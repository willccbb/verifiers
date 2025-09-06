from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import shlex
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Any as AnyType

from datasets import Dataset
from prime_cli.api.sandbox import AsyncSandboxClient, CreateSandboxRequest

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None

@dataclass
class SandboxConfig:
    image: str = "node:22-bookworm"
    cpu_cores: int = 1
    memory_gb: int = 2
    disk_size_gb: int = 10
    start_command: str = "tail -f /dev/null"
    api_key: Optional[str] = None


@dataclass
class MCPServerSpec:
    """
    Specification for a single MCP server we can launch via stdio.
    Example (Everything server):
      name="everything",
      command=["npx", "-y", "@modelcontextprotocol/server-everything", "stdio"]
    """
    name: str
    command: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    transport: str = "stdio"
    discovery_timeout_s: Optional[int] = None
    call_timeout_s: Optional[int] = None


@dataclass
class DiscoveredTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None


# ---------------------------
# Prime Sandbox Management
# ---------------------------

class PrimeSandboxManager:
    """Manage lifecycle + command execution for a Prime sandbox."""

    def __init__(self, config: SandboxConfig):
        self.cfg = config
        self.api_key = self.cfg.api_key or os.getenv("PRIME_API_KEY")
        self.client = AsyncSandboxClient(api_key=self.api_key)
        self.sandbox_id: Optional[str] = None
        self.logger = logging.getLogger("verifiers.envs.MCPToolEnv.SandboxManager")

    async def __aenter__(self) -> "PrimeSandboxManager":
        await self.create()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.cleanup()

    async def create(self) -> str:
        try:
            req = CreateSandboxRequest(
                name=f"mcp-{int(time.time())}",
                docker_image=self.cfg.image,
                cpu_cores=self.cfg.cpu_cores,
                memory_gb=self.cfg.memory_gb,
                disk_size_gb=self.cfg.disk_size_gb,
                start_command=self.cfg.start_command,
            )
            self.logger.info(f"Creating sandbox with image {self.cfg.image}")
            sandbox = await self.client.create(req)
            self.sandbox_id = sandbox.id
            await self.client.wait_for_creation(self.sandbox_id, max_attempts=90)
            self.logger.info(f"Sandbox ready: {self.sandbox_id}")
            return self.sandbox_id
        except Exception:
            self.logger.exception("Failed to create sandbox")
            raise

    async def run(self, command: str) -> Tuple[int, str, str]:
        """Run a shell command inside the sandbox."""
        if not self.sandbox_id:
            raise RuntimeError("Sandbox not created")
        try:
            res = await self.client.execute_command(
                sandbox_id=self.sandbox_id,
                command=command,
            )
            return res.exit_code, res.stdout, res.stderr
        except Exception:
            self.logger.exception("Sandbox command failed")
            raise

    async def cleanup(self) -> None:
        if self.sandbox_id:
            try:
                await self.client.delete(self.sandbox_id)
                self.logger.info(f"Deleted sandbox: {self.sandbox_id}")
            except Exception:
                self.logger.exception("Failed to delete sandbox")
            finally:
                self.sandbox_id = None


# ---------------------------
# MCP Runner Installer
# ---------------------------

MCP_RUNNER_SCRIPT = r"""#!/usr/bin/env python3
import argparse, asyncio, json, os, sys, base64
from typing import Any, Dict, List, Optional

# Official MCP Python SDK (stdio client)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def _json_print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False))

def _join_text(content):
    text_parts = []
    try:
        for c in (content or []):
            if hasattr(c, "text"):
                text_parts.append(c.text)
            elif isinstance(c, dict) and "text" in c:
                text_parts.append(c["text"])
    except Exception:
        pass
    return "\n".join([t for t in text_parts if t is not None])

def _schema_to_dict(schema):
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):  # Pydantic v2
        try:
            return schema.model_dump()
        except Exception:
            pass
    if hasattr(schema, "dict"):        # Pydantic v1
        try:
            return schema.dict()
        except Exception:
            pass
    try:
        return json.loads(json.dumps(schema, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {}

async def _list_tools(server_cmd: List[str], env: Dict[str, str], cwd: Optional[str], timeout_s: float) -> None:
    server_params = StdioServerParameters(command=server_cmd[0], args=server_cmd[1:], env=env, cwd=cwd)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await asyncio.wait_for(session.list_tools(), timeout=timeout_s)
            tools = []
            for t in tools_resp.tools:
                entry = {
                    "name": t.name,
                    "title": getattr(t, "title", None),
                    "description": t.description or f"MCP tool: {t.name}",
                    "inputSchema": _schema_to_dict(getattr(t, "inputSchema", None)),
                }
                out_schema = _schema_to_dict(getattr(t, "outputSchema", None))
                if out_schema:
                    entry["outputSchema"] = out_schema
                tools.append(entry)
            _json_print({"ok": True, "tools": tools})

async def _call_tool(server_cmd: List[str], env: Dict[str, str], cwd: Optional[str], tool_name: str, arguments: Dict[str, Any], timeout_s: float) -> None:
    server_params = StdioServerParameters(command=server_cmd[0], args=server_cmd[1:], env=env, cwd=cwd)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await asyncio.wait_for(session.call_tool(tool_name, arguments=arguments), timeout=timeout_s)
            payload = {
                "structuredContent": getattr(result, "structuredContent", None),
                "text": _join_text(getattr(result, "content", None)),
                "isError": getattr(result, "isError", False),
            }
            _json_print({"ok": True, "result": payload})

def _decode_b64(b64: str) -> Dict[str, Any]:
    return json.loads(base64.b64decode(b64.encode("utf-8")).decode("utf-8"))

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Base64-encoded JSON request")
    g.add_argument("--input-file", help="Path to JSON request file")
    args = ap.parse_args()

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            req = json.load(f)
    else:
        req = _decode_b64(args.input)

    mode = req.get("mode")
    server_cmd = req["server"]["command"]
    env = req["server"].get("env", {})
    cwd = req["server"].get("cwd")
    timeout_s = float(req.get("timeout", 20))

    if mode == "list-tools":
        asyncio.run(_list_tools(server_cmd, env, cwd, timeout_s))
    elif mode == "call-tool":
        tool_name = req["tool_name"]
        arguments = req.get("arguments", {})
        asyncio.run(_call_tool(server_cmd, env, cwd, tool_name, arguments, timeout_s))
    else:
        _json_print({"ok": False, "error": f"unknown mode: {mode}"})

if __name__ == "__main__":
    main()
"""

class MCPBootstrap:
    """Prepare the sandbox with Python MCP SDK and our unified runner."""

    def __init__(self, sandbox: PrimeSandboxManager):
        self.sandbox = sandbox
        self.logger = logging.getLogger("verifiers.envs.MCPToolEnv.MCPBootstrap")

    async def provision(self) -> None:
        # Minimal OS deps: python3, pip, venv, curl
        steps = [
            "apt-get update -y",
            "apt-get install -y --no-install-recommends python3 python3-pip python3-venv curl ca-certificates",
            # Isolated env to avoid PEP 668 issues.
            "python3 -m venv /opt/mcp-venv",
            "/opt/mcp-venv/bin/python -m pip install --upgrade pip",
            '/opt/mcp-venv/bin/pip install --no-cache-dir "mcp[cli]"',
            # quick sanity
            '/opt/mcp-venv/bin/python - << "PY"\nimport sys, mcp\nprint("MCP OK", getattr(mcp, "__version__", "?"), "on", sys.version)\nPY',
            # avoid accidentally using an old runner path:
            "rm -f /usr/local/bin/mcp_runner.py || true",
        ]
        for step in steps:
            self.logger.info(f"[bootstrap] {step}")
            rc, out, err = await self.sandbox.run(step)
            if rc != 0:
                raise RuntimeError(f"Bootstrap step failed: {step}\n{err or out}")

        # Install the runner into the venv
        write_runner = (
            "cat > /opt/mcp-venv/bin/mcp_runner.py <<'PY'\n"
            + MCP_RUNNER_SCRIPT
            + "\nPY\n"
            "chmod +x /opt/mcp-venv/bin/mcp_runner.py"
        )
        rc, out, err = await self.sandbox.run(write_runner)
        if rc != 0:
            raise RuntimeError(f"Failed to install mcp_runner.py: {err or out}")

        # Print Python version in venv (optional sanity)
        await self.sandbox.run('/opt/mcp-venv/bin/python - << "PY"\nimport sys\nprint(\"PY:\", sys.version)\nPY')


# ---------------------------
# Utilities (schema & logging)
# ---------------------------

REMOVED_KEYS = {
    "additionalProperties", "unevaluatedProperties", "deprecated",
    "readOnly", "writeOnly", "$defs", "definitions", "examples",
    "patternProperties", "$schema", "$id",
}

def _clean_schema_for_oai(schema: Any) -> Any:
    """
    Recursively sanitize an MCP JSON Schema so strict validators accept it.
    - Removes keys that commonly trigger strict errors (e.g., additionalProperties).
    - Preserves type, properties, required, enum, default, description, items, oneOf/anyOf/allOf.
    - If properties exists but type is missing, sets type=object.
    """
    if not isinstance(schema, dict):
        return schema

    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k in REMOVED_KEYS:
            continue

        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _clean_schema_for_oai(pv) for pk, pv in v.items()}
            continue

        if k in ("items",):
            out[k] = _clean_schema_for_oai(v)
            continue

        if k in ("oneOf", "anyOf", "allOf") and isinstance(v, list):
            out[k] = [_clean_schema_for_oai(x) for x in v]
            continue

        if isinstance(v, dict):
            out[k] = _clean_schema_for_oai(v)
        elif isinstance(v, list):
            out[k] = [_clean_schema_for_oai(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v

    if "properties" in out and "type" not in out:
        out["type"] = "object"

    return out


def _redact_env(env: Dict[str, str]) -> Dict[str, str]:
    redacted = {}
    for k, v in env.items():
        if any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted[k] = "****"
        else:
            redacted[k] = v
    return redacted


def _prepare_req_file(req: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (path, shell_command) to write JSON request inside the sandbox.
    """
    content = json.dumps(req, ensure_ascii=False)
    fname = f"/tmp/mcp_req_{uuid.uuid4().hex}.json"
    cmd = f"cat > {fname} << 'EOF'\n{content}\nEOF"
    return fname, cmd


# ---------------------------
# MCP Server Manager (stdio)
# ---------------------------

class MCPServerManager:
    """Register MCP servers, discover tools, and call tools via the inline runner."""

    def __init__(
        self,
        sandbox: PrimeSandboxManager,
        tool_discovery_timeout_s: int = 20,
        tool_call_timeout_s: int = 60,
        return_structured: bool = False,
    ):
        self.sandbox = sandbox
        self.logger = logging.getLogger("verifiers.envs.MCPToolEnv.MCPServerManager")
        self.tool_discovery_timeout_s = tool_discovery_timeout_s
        self.tool_call_timeout_s = tool_call_timeout_s
        self.return_structured = return_structured

        self._servers: Dict[str, MCPServerSpec] = {}
        self._tools_by_server: Dict[str, List[DiscoveredTool]] = {}
        # map "<server>:<tool>" -> (server, raw_tool_name)
        self._index: Dict[str, Tuple[str, str]] = {}

    def register(self, spec: MCPServerSpec) -> None:
        self._servers[spec.name] = spec

    async def discover_tools(self, server_name: str) -> List[DiscoveredTool]:
        spec = self._servers[server_name]
        timeout = spec.discovery_timeout_s or self.tool_discovery_timeout_s
        req = {
            "mode": "list-tools",
            "server": {"command": spec.command, "env": spec.env, "cwd": spec.workdir},
            "timeout": timeout,
        }

        payload = json.dumps(req)
        if len(payload) > 64_000:
            path, write_cmd = _prepare_req_file(req)
            await self.sandbox.run(write_cmd)
            cmd = f"/opt/mcp-venv/bin/python /opt/mcp-venv/bin/mcp_runner.py --input-file {shlex.quote(path)}"
        else:
            b64 = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
            cmd = f"/opt/mcp-venv/bin/python /opt/mcp-venv/bin/mcp_runner.py --input {shlex.quote(b64)}"

        self.logger.debug(f"[discover] {server_name} cmd={cmd} env={_redact_env(spec.env)} cwd={spec.workdir}")
        rc, out, err = await self.sandbox.run(cmd)
        if rc != 0:
            raise RuntimeError(f"Tool discovery failed for {server_name}:\n{err or out}")

        try:
            resp = json.loads(out.strip() or "{}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Malformed discovery output for {server_name}: {out[:500]}")

        if not resp.get("ok"):
            raise RuntimeError(f"Discovery error for {server_name}: {resp.get('error')}")

        tools: List[DiscoveredTool] = []
        for t in resp.get("tools", []):
            # Clean schemas to ensure strict validator compatibility
            input_schema = _clean_schema_for_oai(t.get("inputSchema") or {})
            output_schema = t.get("outputSchema")
            if output_schema:
                output_schema = _clean_schema_for_oai(output_schema)

            tools.append(
                DiscoveredTool(
                    name=t["name"],
                    description=t.get("description") or f"MCP tool: {t['name']}",
                    input_schema=input_schema,
                    output_schema=output_schema,
                )
            )

        # Index tools with namespacing "<server>:<tool>"
        self._tools_by_server[server_name] = tools
        for t in tools:
            self._index[f"{server_name}:{t.name}"] = (server_name, t.name)
        self.logger.info(f"Discovered {len(tools)} tools from server '{server_name}'")
        return tools

    async def call_tool(self, namespaced_tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool and return normalized payload {text, structuredContent, isError}."""
        try:
            server_name, tool_name = self._index[namespaced_tool]
        except KeyError:
            return {"error": f"Unknown tool: {namespaced_tool}"}

        spec = self._servers[server_name]
        timeout = spec.call_timeout_s or self.tool_call_timeout_s

        # Allow per-call override via reserved key (e.g., kwargs["_timeout_s"]).
        if isinstance(arguments, dict) and "_timeout_s" in arguments:
            try:
                timeout = int(arguments.pop("_timeout_s")) or timeout
            except Exception:
                pass

        req = {
            "mode": "call-tool",
            "server": {"command": spec.command, "env": spec.env, "cwd": spec.workdir},
            "tool_name": tool_name,
            "arguments": arguments or {},
            "timeout": timeout,
        }

        payload = json.dumps(req)
        if len(payload) > 64_000:
            path, write_cmd = _prepare_req_file(req)
            await self.sandbox.run(write_cmd)
            cmd = f"/opt/mcp-venv/bin/python /opt/mcp-venv/bin/mcp_runner.py --input-file {shlex.quote(path)}"
        else:
            b64 = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
            cmd = f"/opt/mcp-venv/bin/python /opt/mcp-venv/bin/mcp_runner.py --input {shlex.quote(b64)}"

        self.logger.debug(f"[call] {namespaced_tool} timeout={timeout}s env={_redact_env(spec.env)} cwd={spec.workdir}")
        rc, out, err = await self.sandbox.run(cmd)
        if rc != 0:
            return {"error": f"Tool call failed (exit {rc}): {err or out}"}

        try:
            resp = json.loads(out.strip() or "{}")
        except json.JSONDecodeError:
            return {"error": f"Malformed tool output: {out[:500]}"}

        if not resp.get("ok"):
            return {"error": resp.get("error")}

        return resp.get("result", {"text": "", "structuredContent": None, "isError": False})

    def list_registered(self) -> List[str]:
        return list(self._servers.keys())

    def tools_index(self) -> Dict[str, Tuple[str, str]]:
        return dict(self._index)


# ---------------------------
# Tool wrappers for verifiers
# ---------------------------

def _normalize_tool_func_name(namespaced: str) -> str:
    # Convert "<server>:<tool>" to a safe python identifier
    safe = namespaced.replace(":", "__").replace("/", "_")
    return f"mcp__{safe}"


def create_tool_wrapper(
    namespaced_tool: str,
    tool_meta: DiscoveredTool,
    server_mgr: MCPServerManager,
    return_structured: bool = False,
) -> Callable:
    """
    Create a sync wrapper that internally uses asyncio to call MCP tool.
    Returns plaintext by default (agent-safe); can return structured-only when allowed.
    """

    # Sanitize/normalize schema for signature + optional runtime validation
    cleaned_schema = _clean_schema_for_oai(tool_meta.input_schema or {})
    props: Dict[str, Any] = cleaned_schema.get("properties", {}) or {}
    required = set(cleaned_schema.get("required", []) or [])

    validator = None
    if jsonschema and cleaned_schema.get("type") == "object":
        try:
            validator = jsonschema.Draft202012Validator(cleaned_schema)
        except Exception:
            validator = None

    async def _async_call(kwargs: Dict[str, Any]) -> str:
        # Optional input validation (friendly error messages)
        if validator:
            errors = sorted(validator.iter_errors(kwargs), key=lambda e: e.path)
            if errors:
                msgs = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]
                return "Input validation failed:\n- " + "\n- ".join(msgs)

        result = await server_mgr.call_tool(namespaced_tool, kwargs or {})
        if "error" in result:
            return f"Tool error: {result['error']}"

        sc = result.get("structuredContent")
        text = result.get("text") or ""

        if sc is not None:
            if return_structured or server_mgr.return_structured:
                # Return strictly the structured object as JSON text (agents may validate)
                try:
                    return json.dumps(sc, ensure_ascii=False)
                except Exception:
                    # Fallback to embedding both if non-serializable
                    return json.dumps({"structured": sc, "text": text}, ensure_ascii=False)
            # Default: agent-safe
            return json.dumps({"structured": sc, "text": text}, ensure_ascii=False)

        return text

    params: List[inspect.Parameter] = []
    param_names: List[str] = []

    for prop_name, prop_spec in props.items():
        param_names.append(prop_name)
        if prop_name in required:
            # Required parameter - no default
            params.append(
                inspect.Parameter(
                    prop_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=AnyType,
                )
            )
        else:
            # Optional parameter - default to None
            params.append(
                inspect.Parameter(
                    prop_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=AnyType,
                )
            )

    sig = inspect.Signature(params)

    def wrapper(*args, **kwargs) -> str:
        # Map positional args to kwargs by name, then merge keyword args
        call_kwargs: Dict[str, Any] = {}
        for i, arg in enumerate(args):
            if i < len(param_names):
                call_kwargs[param_names[i]] = arg

        # Filter out None values for optional params
        for key, value in kwargs.items():
            if key in param_names and value is not None:
                call_kwargs[key] = value
            elif key == "_timeout_s":
                # allow the reserved override through
                call_kwargs[key] = value

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Run in a worker thread to avoid interfering with an active loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    fut = ex.submit(asyncio.run, _async_call(call_kwargs))
                    return fut.result()
            else:
                return loop.run_until_complete(_async_call(call_kwargs))
        except RuntimeError:
            # No running loop
            return asyncio.run(_async_call(call_kwargs))

    # Name, signature and docstring
    wrapper.__name__ = _normalize_tool_func_name(namespaced_tool)
    wrapper.__signature__ = sig  # Set the dynamic signature

    # Build a docstring from cleaned schema (include defaults & enums if present)
    doc = [tool_meta.description or f"MCP tool: {namespaced_tool}", "", "Args:"]
    for p, spec in props.items():
        typ = spec.get("type", "any")
        desc = spec.get("description", p)
        default = spec.get("default", None)
        enum = spec.get("enum", None)
        frag = f"  {p} ({typ}){' [required]' if p in required else ''}: {desc}"
        if default is not None:
            frag += f" (default: {default!r})"
        if enum:
            frag += f" (one of: {', '.join(map(str, enum))})"
        doc.append(frag)
    wrapper.__doc__ = "\n".join(doc) if props else (tool_meta.description or "MCP tool")

    return wrapper


# ---------------------------
# MCP Tool Environment
# ---------------------------

class MCPToolEnv(vf.ToolEnv):
    """
    Tool environment that:
      - spins up a Prime sandbox (Node image),
      - bootstraps MCP Python SDK and a small runner,
      - registers one or more MCP servers (stdio),
      - discovers tools and exposes them to verifiers,
      - **cleans up deterministically after execution** (auto_cleanup).

    Can be used as a context manager to ensure proper cleanup:
        async with MCPToolEnv(...) as env:
            # use env
            pass
        # sandbox is automatically cleaned up
    """

    def __init__(
        self,
        mcp_servers: List[Dict[str, Any]],
        sandbox_config: Optional[Dict[str, Any]] = None,
        dataset: Optional[Dataset] = None,
        parser: Optional[vf.Parser] = None,
        rubric: Optional[vf.ToolRubric] = None,
        max_turns: int = 10,
        prime_api_key: Optional[str] = None,
        return_structured: bool = False,
        auto_cleanup: bool = True,            # <--- NEW: deterministic cleanup
        **kwargs,
    ):
        cfg = SandboxConfig(**(sandbox_config or {}), api_key=prime_api_key or (sandbox_config or {}).get("api_key"))
        self.sandbox_mgr = PrimeSandboxManager(cfg)
        self.bootstrap = MCPBootstrap(self.sandbox_mgr)
        self.server_mgr = MCPServerManager(
            self.sandbox_mgr,
            return_structured=return_structured,
        )

        self.return_structured = return_structured
        self.auto_cleanup = auto_cleanup

        self._initialized = False
        self._cleanup_done = False
        self._active_rollouts = 0
        self._counter_lock = asyncio.Lock()

        self.logger = logging.getLogger("verifiers.envs.MCPToolEnv")

        super().__init__(tools=[], dataset=dataset, parser=parser, rubric=rubric, max_turns=max_turns, **kwargs)

        self._server_specs: List[MCPServerSpec] = [
            MCPServerSpec(
                name=s["name"],
                command=s["command"],
                env=s.get("env", {}),
                workdir=s.get("workdir"),
                transport=s.get("transport", "stdio"),
                discovery_timeout_s=s.get("discovery_timeout_s"),
                call_timeout_s=s.get("call_timeout_s"),
            )
            for s in mcp_servers
        ]

    async def _initialize(self) -> None:
        if self._initialized:
            return

        # 1) Bring up sandbox
        await self.sandbox_mgr.create()

        # 2) Bootstrap Python MCP SDK + our runner
        await self.bootstrap.provision()

        # 3) Register servers & discover tools
        all_tools: List[Callable] = []
        for spec in self._server_specs:
            self.server_mgr.register(spec)
            tools = await self.server_mgr.discover_tools(spec.name)
            for t in tools:
                namespaced = f"{spec.name}:{t.name}"
                tool_fn = create_tool_wrapper(namespaced, t, self.server_mgr, return_structured=self.return_structured)
                all_tools.append(tool_fn)

        # 4) Surface tools to verifiers ToolEnv
        self.tools = all_tools
        from verifiers.utils.tool_utils import convert_func_to_oai_tool
        self.oai_tools = [convert_func_to_oai_tool(fn) for fn in self.tools]
        self.tool_map = {fn.__name__: fn for fn in self.tools}

        self._initialized = True
        self.logger.info(f"Initialized MCPToolEnv with {len(all_tools)} tools from {len(self._server_specs)} server(s)")

    async def rollout(self, *args, **kwargs) -> Tuple[vf.Messages, vf.State]:
        # Ensure initialized once
        await self._initialize()

        # Track concurrent rollouts
        async with self._counter_lock:
            self._active_rollouts += 1

        try:
            return await super().rollout(*args, **kwargs)
        finally:
            # Decrement and possibly cleanup *after* the last rollout finishes
            do_cleanup = False
            async with self._counter_lock:
                self._active_rollouts -= 1
                if self.auto_cleanup and not self._cleanup_done and self._active_rollouts <= 0:
                    do_cleanup = True
                    self._cleanup_done = True
            if do_cleanup:
                try:
                    await self.cleanup()
                except Exception:
                    self.logger.exception("Auto-cleanup failed")

    async def __aenter__(self) -> "MCPToolEnv":
        """Async context manager entry - initialize the environment."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup the sandbox."""
        await self.cleanup()

    def __del__(self):
        """Best-effort silent cleanup on GC (fallback only)."""
        try:
            if getattr(self, "sandbox_mgr", None) and getattr(self.sandbox_mgr, "sandbox_id", None):
                # Avoid warnings; attempt best-effort cleanup if still open
                async def _finalize():
                    try:
                        await self.cleanup()
                    except Exception:
                        pass
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import threading
                        threading.Thread(target=lambda: asyncio.run(_finalize()), daemon=True).start()
                    else:
                        asyncio.run(_finalize())
                except Exception:
                    pass
        except Exception:
            pass

    async def cleanup(self) -> None:
        """Cleanup the sandbox and resources (idempotent)."""
        try:
            await self.sandbox_mgr.cleanup()
            self._cleanup_done = True
            self.logger.info("MCPToolEnv cleanup completed")
        except Exception:
            self.logger.exception("Cleanup failed")


# ---------------------------
# Convenience Loader
# ---------------------------

def load_environment(
    mcp_launch_commands: List[Dict[str, Any]],
    dataset_name: str = "gsm8k",
    split: str = "train",
    num_examples: int = 100,
    sandbox_config: Optional[Dict[str, Any]] = None,
    max_turns: int = 10,
    prime_api_key: Optional[str] = None,
    return_structured: bool = False,
    auto_cleanup: bool = True,
    **kwargs
) -> MCPToolEnv:
    """
    Load MCPToolEnv with specified MCP server configurations.

    Args:
        mcp_launch_commands: List of MCP server configs (see MCPServerSpec). Example:
            [{"name": "everything", "command": ["npx", "-y", "@modelcontextprotocol/server-everything", "stdio"]}]
        dataset_name: Dataset to load via `load_example_dataset`
        split: Dataset split
        num_examples: Number of examples
        sandbox_config: Overrides for Node sandbox
        max_turns: Max conversation turns
        prime_api_key: Prime API key (optional)
        return_structured: If True, wrappers return structured JSON only (agents must accept object outputs).
                           Default False returns agent-safe strings.
        auto_cleanup: If True (default), the sandbox is cleaned up automatically after the last rollout completes.
    """
    dataset = load_example_dataset(dataset_name, split=split)
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    parser = vf.Parser()
    rubric = vf.ToolRubric(tools=[])

    env = MCPToolEnv(
        mcp_servers=mcp_launch_commands,
        sandbox_config=sandbox_config,
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        prime_api_key=prime_api_key,
        return_structured=return_structured,
        auto_cleanup=auto_cleanup,
        **kwargs,
    )
    return env
