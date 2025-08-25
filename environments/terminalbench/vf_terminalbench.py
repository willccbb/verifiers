"""
Terminal-Bench environment for Verifiers

This environment reuses Terminal-Bench's native harness components
to run tasks in Docker via docker-compose and a tmux session, exposing a single
`execute_commands` tool for the agent. It avoids duplicating container logic.
"""

import atexit
import asyncio
import importlib
import os
import types
import signal
import sys
import tempfile
import time
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64
import builtins as _builtins
import threading

from datasets import Dataset

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv
from verifiers.rubrics.rubric import RolloutScores

# Import only the specific terminal-bench modules we need, without importing the
# package-level __init__ (which pulls heavy agent deps).
try:
    from terminal_bench.handlers.trial_handler import Task, TaskPaths, TrialHandler  # type: ignore
    from terminal_bench.terminal.terminal import Terminal  # type: ignore
    from terminal_bench.terminal.tmux_session import TmuxSession  # type: ignore
    from terminal_bench.terminal.docker_compose_manager import DockerComposeManager  # type: ignore
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    tb_root = repo_root / "terminal-bench"
    # Prefer dependency-managed install. Allow local dev fallback ONLY if TB_DEV_LOCAL=1
    if os.getenv("TB_DEV_LOCAL") == "1":
        # Create a lightweight package stub to avoid executing terminal_bench/__init__.py
        pkg_dir = tb_root / "terminal_bench"
        if not pkg_dir.exists():
            raise ModuleNotFoundError(
                f"terminal-bench source not found at {pkg_dir}. Please install the dependency or set TB_DEV_LOCAL=0."
            )

        if "terminal_bench" not in sys.modules:
            stub = types.ModuleType("terminal_bench")
            stub.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
            sys.modules["terminal_bench"] = stub

        # Import needed submodules normally; they'll use the stub's __path__
        trial_handler_mod = importlib.import_module(
            "terminal_bench.handlers.trial_handler"
        )
        terminal_mod = importlib.import_module("terminal_bench.terminal.terminal")
        tmux_mod = importlib.import_module("terminal_bench.terminal.tmux_session")
        dcm_mod = importlib.import_module(
            "terminal_bench.terminal.docker_compose_manager"
        )

        Task = getattr(trial_handler_mod, "Task")
        TaskPaths = getattr(trial_handler_mod, "TaskPaths")
        TrialHandler = getattr(trial_handler_mod, "TrialHandler")
        Terminal = getattr(terminal_mod, "Terminal")
        TmuxSession = getattr(tmux_mod, "TmuxSession")
        DockerComposeManager = getattr(dcm_mod, "DockerComposeManager")
    else:
        raise ModuleNotFoundError(
            "terminal_bench is not installed. Please add it as a dependency (see pyproject) "
            "and use Python 3.12+. Repo: https://github.com/laude-institute/terminal-bench"
        )


# Precompiled regex for ANSI escape sequences to improve performance
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# Thread-local storage for per-tool-call context (enables parallelism without races)
THREAD_LOCAL = threading.local()


# Timestamped print wrapper (enabled by default; set TB_TS_LOGS=0 to disable)
def _ts() -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    except Exception:
        return ""


if os.getenv("TB_TS_LOGS", "1") == "1":
    _orig_print = _builtins.print

    def print(*args, **kwargs):  # type: ignore
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        file = kwargs.get("file", None)
        flush = kwargs.get("flush", False)
        try:
            msg = sep.join(str(a) for a in args)
        except Exception:
            msg = " ".join(["<non-str-arg>"] * len(args))
        _orig_print(f"[{_ts()}] {msg}", end=end, file=file, flush=flush)


# Concurrency controls (no legacy fallbacks)
ROLLOUT_CONCURRENCY = int(os.getenv("TB_ROLLOUT_CONCURRENCY", "1"))


class _TerminalContext:
    """Holds Terminal-Bench resources for a single task/session."""

    def __init__(self, task_path: Path, output_root: Path):
        # Create a TrialHandler to leverage naming, paths, and metadata
        trial_name = f"verifiers.1-of-1.{int(time.time())}.{uuid.uuid4().hex[:6]}"
        self.trial_handler = TrialHandler(
            trial_name=trial_name,
            input_path=task_path,
            output_path=output_root,
        )
        print(
            f"[TERMINALCTX] Init trial_name={trial_name} task_id={self.trial_handler.task_paths.input_path.name}"
        )

        # Initialize Terminal using Terminal-Bench's compose manager
        disable_recording = self.trial_handler.task.disable_asciinema
        # Allow controlling rebuild/cleanup behavior via env to speed up local runs
        env_no_rebuild = os.getenv("TB_NO_REBUILD", "0") == "1"
        env_cleanup = os.getenv("TB_CLEANUP", "0") == "1"
        self.terminal = Terminal(
            client_container_name=self.trial_handler.client_container_name,
            client_image_name=self.trial_handler.client_image_name,
            docker_image_name_prefix=self.trial_handler.docker_image_name_prefix,
            docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
            sessions_logs_path=self.trial_handler.trial_paths.sessions_path,
            agent_logs_path=self.trial_handler.trial_paths.agent_logging_dir,
            commands_path=self.trial_handler.trial_paths.commands_path,
            no_rebuild=env_no_rebuild,
            cleanup=env_cleanup,
            livestream=False,
            disable_recording=disable_recording,
        )

        self.session: Optional[TmuxSession] = None

    def start(self) -> None:
        t0 = time.time()
        print("[TERMINALCTX] Starting docker compose and container...")
        self.terminal.start()
        print(
            f"[TERMINALCTX] Container started in {time.time() - t0:.2f}s; creating tmux session..."
        )
        # Run as configured user for agent session
        self.session = self.terminal.create_session(
            "agent", is_active_stream=False, as_configured_user=True
        )
        print("[TERMINALCTX] Tmux session 'agent' created.")
        # Best-effort: ensure shell prompt is responsive before first use
        try:
            if self.session:
                self.session.send_keys(
                    ["echo __TB_SESSION_READY__", "Enter"],
                    block=True,
                    max_timeout_sec=10,
                )
                # Drain any immediate output produced during startup (visible screen only)
                _ = self.session.capture_pane(capture_entire=False)
        except Exception:
            # Non-fatal; continue even if readiness probe fails
            pass

    def stop(self) -> None:
        try:
            print("[TERMINALCTX] Stopping terminal and docker compose...")
            self.terminal.stop()
            print("[TERMINALCTX] Terminal stopped.")
        except Exception:
            pass

    def send_and_capture(self, command: str, timeout: float) -> Tuple[bool, str]:
        if not self.session:
            raise RuntimeError("Terminal session not started")

        # Execute command in a blocking manner and capture incremental output
        try:
            print(f"[TERMINALCTX] Executing command (timeout={timeout}s): {command}")
            t0 = time.time()
            self.session.send_keys(
                [command, "Enter"], block=True, max_timeout_sec=timeout
            )
        except TimeoutError:
            # Attempt to gracefully interrupt and capture whatever is available
            try:
                # Send Ctrl-C to stop the running foreground command
                self.session.send_keys(["C-c"], block=False, max_timeout_sec=1)
            except Exception:
                pass
            try:
                output = self.session.capture_pane(capture_entire=False)
            except Exception:
                output = ""
            print(
                f"[TERMINALCTX] Command timed out after {timeout}s; captured {len(output)} chars."
            )
            return (
                False,
                output + f"\n[terminalbench] Command timed out after {timeout}s",
            )

        output = self.session.capture_pane(capture_entire=False)
        print(
            f"[TERMINALCTX] Command completed in {time.time() - t0:.2f}s; captured {len(output)} chars."
        )

        # Heuristic: consider success if no clear error keywords appear in last pane
        pane_text = self.session.capture_pane(capture_entire=False)
        failed = any(
            k in pane_text for k in ["command not found", "Traceback", "ERROR"]
        )  # noqa: E501
        return (not failed), output

    def run_block(self, commands_block: str, timeout: float) -> Tuple[bool, str]:
        """Execute a multi-line block via a single base64-decoded script.

        This avoids per-line splitting and preserves here-documents safely.
        """
        if not self.session:
            raise RuntimeError("Terminal session not started")

        encoded = base64.b64encode(commands_block.encode("utf-8")).decode("ascii")
        # Do not exit the shell; print a status marker we can parse
        status_marker = "__VF_STATUS__"
        print(
            f"[TERMINALCTX] Running block of {len(commands_block)} chars (timeout={timeout}s)."
        )
        one_liner = (
            "tmpfile=$(mktemp /tmp/vf_cmd.XXXXXX.sh) && "
            f"echo '{encoded}' | base64 -d > \"$tmpfile\" && "
            'chmod +x "$tmpfile" && bash "$tmpfile"; status=$?; '
            'rm -f "$tmpfile"; echo ' + status_marker + ":$status"
        )

        t0 = time.time()
        ok, out = self.send_and_capture(one_liner, timeout)
        print(
            f"[TERMINALCTX] Block finished in {time.time() - t0:.2f}s; ok={ok}; output_len={len(out)}"
        )
        # Determine success from explicit status marker if present
        match = re.search(r"__VF_STATUS__:(\d+)", out)
        if match:
            exit_code_str = match.group(1)
            try:
                exit_code = int(exit_code_str)
                ok = ok and (exit_code == 0)
            except ValueError:
                pass
        return ok, out

    def run_tests(self, timeout: float) -> Tuple[bool, str]:
        if not self.session:
            raise RuntimeError("Terminal session not started")

        # Copy tests and run-tests.sh similar to Harness
        paths = [self.trial_handler.task_paths.run_tests_path]
        if self.trial_handler.task_paths.test_dir.exists():
            paths.append(self.trial_handler.task_paths.test_dir)

        self.terminal.copy_to_container(
            paths=paths,
            container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
        )

        # Follow Harness behavior: optionally use a separate shell for tests
        test_session = self.session
        if not self.trial_handler.task.run_tests_in_same_shell:
            test_session = self.terminal.create_session(
                "tests", is_active_stream=False, as_configured_user=False
            )

        # Execute tests
        test_script_name = self.trial_handler.task_paths.run_tests_path.name
        test_cmd = f"bash {DockerComposeManager.CONTAINER_TEST_DIR / test_script_name}"
        try:
            test_session.send_keys(
                [test_cmd, "Enter"], block=True, max_timeout_sec=timeout
            )  # noqa: E501
        except TimeoutError:
            return False, f"[terminalbench] Test execution timed out after {timeout}s"

        post_test = test_session.capture_pane(capture_entire=True)
        return ("PASSED" in post_test and "FAILED" not in post_test), post_test


class TerminalTaskExecutor:
    """Manages Terminal-Bench terminals for tasks, one per task."""

    def __init__(self):
        self.contexts: Dict[str, _TerminalContext] = {}
        self.output_root = Path(tempfile.mkdtemp(prefix="terminalbench_vf_"))
        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self) -> None:
        atexit.register(self.cleanup)

        if os.getenv("TB_HANDLE_SIGNALS", "0") == "1":

            def _handler(signum, frame):
                print(f"\nReceived signal {signum}, cleaning up...")
                self.cleanup()
                sys.exit(0)

            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)

    def get_context(self, task_id: str, task_path: Path) -> _TerminalContext:
        if task_id not in self.contexts:
            ctx = _TerminalContext(task_path=task_path, output_root=self.output_root)
            ctx.start()
            self.contexts[task_id] = ctx
        return self.contexts[task_id]

    def cleanup_context(self, task_id: str) -> None:
        ctx = self.contexts.pop(task_id, None)
        if ctx:
            ctx.stop()

    def cleanup(self) -> None:
        for tid in list(self.contexts.keys()):
            self.cleanup_context(tid)
        try:
            import shutil

            shutil.rmtree(self.output_root, ignore_errors=True)
        except Exception:
            pass


def load_terminalbench_dataset(
    tasks_root: Optional[Path] = None,
    num_examples: int = -1,
) -> Dataset:
    """Build a lightweight dataset from local Terminal-Bench tasks.

    Returns a HF-style Dataset of entries with minimal info needed by ToolEnv.
    """
    # Default to the checked-out terminal-bench tasks folder
    if tasks_root is None:
        repo_root = Path(__file__).resolve().parents[2]
        tasks_root = repo_root / "terminal-bench" / "tasks"

    if not tasks_root.exists():
        raise RuntimeError(f"Terminal-Bench tasks directory not found at {tasks_root}")

    entries: List[Dict[str, Any]] = []
    tasks = sorted([p for p in tasks_root.iterdir() if p.is_dir()])

    # Prefer lightweight tasks for quick smoke tests
    preferred_order = [
        "hello-world",
        "vim-terminal-task",
        "simple-web-scraper",
    ]
    preferred = [p for p in tasks if p.name in preferred_order]
    others = [p for p in tasks if p.name not in preferred_order]
    tasks = preferred + others

    if num_examples > 0:
        tasks = tasks[:num_examples]

    for task_path in tasks:
        task_id = task_path.name
        paths = TaskPaths(task_path)
        task = Task.from_yaml(paths.task_config_path)

        # Use Terminal-Bench's task instruction verbatim as the prompt
        prompt = task.instruction

        guidance = (
            "You are inside a Linux container for this task. A tool named "
            "execute_commands is available; use it to run shell commands to "
            "complete the task. Work in /app. Use non-interactive commands, "
            "verify results, and keep outputs concise."
        )

        entries.append(
            {
                "prompt": [
                    {"role": "system", "content": guidance},
                    {"role": "user", "content": prompt},
                ],
                "answer": "",
                "info": {
                    "task_id": task_id,
                    "task_path": str(task_path),
                    "max_agent_timeout_sec": task.max_agent_timeout_sec,
                    "max_test_timeout_sec": task.max_test_timeout_sec,
                },
            }
        )

    return Dataset.from_list(entries)


def load_environment(
    dataset_name: str = "local-terminal-bench",  # unused, retained for compatibility
    split: str = "test",  # unused
    num_examples: int = -1,
) -> vf.ToolEnv:
    """Load Terminal-Bench environment backed by terminal_bench primitives."""
    dataset = load_terminalbench_dataset(num_examples=num_examples)

    # Initialize task executor
    executor = TerminalTaskExecutor()

    # Create parser
    def extract_commands(completion: str) -> str:
        """Extract shell commands from function calls (not used for evaluation)"""
        return ""  # Not used since we use ToolEnv's built-in function calling

    parser = vf.Parser(extract_fn=extract_commands)

    # Define the execute_commands function for ToolEnv
    def execute_commands(commands: List[str], reasoning: str = "") -> str:
        """Execute shell commands in the terminal environment.

        Args:
            commands: Array of shell commands to execute
            reasoning: Optional explanation of what these commands do

        Returns:
            Result of command execution including output
        """
        print("[TERMINALBENCH] 🖥️  execute_commands called")
        print(f"[TERMINALBENCH]   Commands type: {type(commands)}")
        print(f"[TERMINALBENCH]   Commands: {commands}")
        print(f"[TERMINALBENCH]   Reasoning: {reasoning}")

        if not commands:
            return "❌ ERROR: No commands provided. You must provide at least one command to execute."

        # Get task context from thread-local state first, fallback to function attrs
        task_id = (
            getattr(THREAD_LOCAL, "task_id", None) or execute_commands._current_task_id
        )
        task_path_str = (
            getattr(THREAD_LOCAL, "task_path", None)
            or execute_commands._current_task_path
        )

        print(f"[TERMINALBENCH]   Current task_id: {task_id}")
        print(f"[TERMINALBENCH]   Task path set: {bool(task_path_str)}")

        if not task_id or not task_path_str:
            return "❌ ERROR: Terminal environment not properly initialized."

        try:
            # Handle both string and array inputs for commands
            if isinstance(commands, str):
                commands_str = commands
            elif isinstance(commands, list):
                commands_str = "\n".join(str(cmd) for cmd in commands)
            else:
                return f"❌ ERROR: Commands must be a string or array of strings, got {type(commands)}"

            # Get or create terminal context for this task
            task_path = Path(task_path_str)
            ctx = executor.get_context(task_id, task_path)

            # Determine timeout per command block with rollout-wide budget
            info = (
                getattr(THREAD_LOCAL, "info", None)
                or getattr(execute_commands, "_current_info", {})
                or {}
            )
            # Base per-call cap
            base_timeout: Optional[float]
            env_timeout = os.getenv("TB_CMD_TIMEOUT_SEC")
            try:
                base_timeout = float(env_timeout) if env_timeout else None
            except Exception:
                base_timeout = None
            if base_timeout is None:
                try:
                    base_timeout = float(info.get("max_agent_timeout_sec", 180))
                except Exception:
                    base_timeout = 180.0

            # Remaining rollout budget if deadline set
            remaining: Optional[float] = None
            deadline = getattr(THREAD_LOCAL, "deadline", None)
            if isinstance(deadline, (float, int)):
                remaining = max(0.0, float(deadline) - time.time())

            if remaining is not None and remaining <= 0.0:
                return "❌ ERROR: Agent time budget exhausted; not executing further commands."

            if remaining is not None:
                timeout_sec = max(1.0, min(base_timeout, remaining))
            else:
                timeout_sec = float(base_timeout)

            # Execute as a single block to preserve heredocs and multi-line structure
            success, output = ctx.run_block(commands_str, timeout=timeout_sec)

            # Sanitize terminal output to avoid overly large or non-JSON-safe payloads
            def _sanitize_output(s: str) -> str:
                # Replace non-printable characters except tab/newline/carriage return
                return "".join(
                    ch if (ch.isprintable() or ch in "\t\n\r") else "�" for ch in s
                )

            # Truncate early, then strip ANSI and sanitize for performance
            max_output_length = 8000
            raw = output
            if len(raw) > max_output_length:
                raw = raw[:max_output_length]
                raw += f"\n\n... [Output truncated. Total length: {len(output)} characters]"
            raw = ANSI_ESCAPE_RE.sub("", raw)
            truncated_output = _sanitize_output(raw)

            # Format response
            result = "Command(s) executed"
            if reasoning:
                result += f" ({reasoning})"
            result += f":\n\n```bash\n{commands_str}\n```\n\n"

            if success:
                result += f"✅ **Success**\n\nOutput:\n```\n{truncated_output}\n```"
            else:
                result += f"❌ **Failed**\n\nOutput:\n```\n{truncated_output}\n```"

            return result

        except Exception as e:
            return f"❌ Execution error: {str(e)}"

    # Set up function attributes that will be set during conversation
    execute_commands._current_task_id = None
    execute_commands._current_task_path = None
    execute_commands._current_info = None

    # Define rubric functions for evaluation
    def task_completion_score(completion, info, parser, state) -> float:
        """Evaluate task completion by running the final tests"""
        print("\n⚖️  EVALUATING TASK COMPLETION ⚖️")

        try:
            # If we already evaluated at rollout end, reuse stored results to avoid re-spinning containers
            try:
                if isinstance(state, dict) and state.get("_tb_evaluated", False):
                    print("Using cached Terminal-Bench test results from rollout")
                    task_id = info.get("task_id", "<unknown>")
                    task_path = (
                        Path(info.get("task_path", ""))
                        if info.get("task_path")
                        else None
                    )  # type: ignore
                    print(f"Task ID: {task_id}")
                    print(f"Task path: {task_path}")

                    success = bool(state.get("terminalbench_parsed_success", False))
                    post_test_pane = str(state.get("terminalbench_test_output", ""))

                    # Diagnostics output similar to normal path
                    def _strip_ansi(s: str) -> str:
                        return ANSI_ESCAPE_RE.sub("", s)

                    def _sanitize(s: str) -> str:
                        s = _strip_ansi(s)
                        return "".join(
                            ch if (ch.isprintable() or ch in "\t\n\r") else "�"
                            for ch in s
                        )

                    sanitized_pane = _sanitize(post_test_pane)
                    pane_lines = sanitized_pane.splitlines()
                    tail_lines = 120
                    tail_text = "\n".join(pane_lines[-tail_lines:])
                    if len(tail_text) > 6000:
                        tail_text = tail_text[-6000:]

                    print("\n🧪 Test run status:")
                    print(
                        f" - Runner indicated ok: {bool(state.get('terminalbench_ran_ok', False))}"
                    )
                    if state.get("terminalbench_parsed_results") is not None:
                        try:
                            print(
                                f" - Parsed results: {state.get('terminalbench_parsed_results')}"
                            )
                        except Exception:
                            pass
                    print("----- Test output (tail) -----")
                    print(tail_text)
                    print("----- End test output -----\n")

                    # Agent commands tail if available
                    if state.get("terminalbench_commands_log_tail"):
                        print("📜 Agent commands log (tail):")
                        print("----- Commands (tail) -----")
                        print(str(state.get("terminalbench_commands_log_tail")))
                        print("----- End commands -----\n")

                    print("\n📋 FINAL EVALUATION RESULT:")
                    print(f"Tests passed: {success}")
                    print(f"Score: {1.0 if success else 0.0}")
                    if not success:
                        print("❌ Task failed Terminal-Bench tests")
                    else:
                        print("✅ Task passed all Terminal-Bench tests!")

                    # Context was already cleaned up at rollout end
                    return 1.0 if success else 0.0
            except Exception as e_cached:
                print(
                    f"Warning: failed to use cached test results, falling back to fresh eval: {e_cached}"
                )

            task_id = info["task_id"]
            task_path = Path(info["task_path"])  # type: ignore

            print(f"Task ID: {task_id}")
            print(f"Task path: {task_path}")

            # Always ensure a context exists/reused for this task
            try:
                os.environ.setdefault("TB_NO_REBUILD", "1")
                os.environ.setdefault("TB_CLEANUP", "0")
                ctx = executor.get_context(task_id, task_path)
            except Exception as e:
                print(f"Failed to create context for {task_id}: {e}")
                return 0.0
            print(f"✅ Ready context for task {task_id}")

            # Run the final tests inside the container
            print("🔬 Running Terminal-Bench test suite...")
            ran_ok, post_test_pane = ctx.run_tests(
                timeout=float(info["max_test_timeout_sec"])  # type: ignore
            )

            # Parse results using Terminal-Bench's parser (1:1 behavior)
            parsed = None
            try:
                parsed = ctx.trial_handler.parser.parse(post_test_pane)
                all_passed = (
                    parsed is not None
                    and len(parsed) > 0
                    and all("PASSED" in str(v) for v in parsed.values())
                )
                success = bool(all_passed)
            except Exception as pe:
                print(f"Parser error: {pe}")
                success = False

            # Provide detailed diagnostics
            def _strip_ansi(s: str) -> str:
                return ANSI_ESCAPE_RE.sub("", s)

            def _sanitize(s: str) -> str:
                s = _strip_ansi(s)
                return "".join(
                    ch if (ch.isprintable() or ch in "\t\n\r") else "�" for ch in s
                )

            sanitized_pane = _sanitize(post_test_pane)
            # Show the last 120 lines (or up to 6000 chars) for brevity
            pane_lines = sanitized_pane.splitlines()
            tail_lines = 120
            tail_text = "\n".join(pane_lines[-tail_lines:])
            if len(tail_text) > 6000:
                tail_text = tail_text[-6000:]

            print("\n🧪 Test run status:")
            print(f" - Runner indicated ok: {ran_ok}")
            print(f" - Parsed results available: {parsed is not None}")
            if parsed is not None:
                try:
                    print(f" - Parsed results: {parsed}")
                except Exception:
                    pass
            print("----- Test output (tail) -----")
            print(tail_text)
            print("----- End test output -----\n")

            # Show the agent's executed commands (if any) for this task
            try:
                commands_log_path = ctx.trial_handler.trial_paths.commands_path
                if commands_log_path.exists():
                    try:
                        log_text = commands_log_path.read_text(errors="replace")
                        log_lines = log_text.splitlines()
                        log_tail = "\n".join(log_lines[-80:])
                        print("📜 Agent commands log (tail):")
                        print(str(commands_log_path))
                        print("----- Commands (tail) -----")
                        print(log_tail)
                        print("----- End commands -----\n")
                    except Exception as le:
                        print(f"Warning: failed to read commands log: {le}")
                else:
                    print("📜 Agent commands log: (no commands log file found)")
            except Exception as le:
                print(f"Warning: could not access commands log path: {le}")

            print("\n📋 FINAL EVALUATION RESULT:")
            print(f"Tests passed: {success}")
            print(f"Score: {1.0 if success else 0.0}")

            if not success:
                print("❌ Task failed Terminal-Bench tests")
            else:
                print("✅ Task passed all Terminal-Bench tests!")

            # Clean up after testing
            print(f"🧹 Cleaning up terminal for {task_id}")
            executor.cleanup_context(task_id)

            return 1.0 if success else 0.0

        except Exception as e:
            print(f"❌ Error during task evaluation: {e}")
            print(f"Exception type: {type(e)}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

            # Clean up even if evaluation failed
            try:
                if task_id in executor.contexts:
                    print(f"🧹 Cleaning up terminal for {task_id} after error")
                    executor.cleanup_context(task_id)
            except Exception as cleanup_e:
                print(f"Warning: Failed to cleanup container after error: {cleanup_e}")

            return 0.0

    # Create rubric (parallelize evaluation/tests with a separate cap)
    class ParallelTestRubric(vf.Rubric):
        def __init__(self, parser, max_parallel_tests: int):
            super().__init__(
                funcs=[task_completion_score],
                weights=[1.0],
                parser=parser,
                parallelize_scoring=False,
            )
            self._max_parallel_tests = max(1, int(max_parallel_tests))

        async def score_rollouts(
            self,
            prompts,
            completions,
            answers,
            states,
            tasks,
            infos,
            **kwargs,
        ) -> RolloutScores:
            sem = asyncio.Semaphore(self._max_parallel_tests)

            async def run_one(p, c, a, s, t, i):
                async with sem:
                    # Offload blocking test execution to a thread
                    return await asyncio.to_thread(
                        task_completion_score, c, i, self.parser, s
                    )

            coros = [
                run_one(p, c, a, s, t, i)
                for p, c, a, s, t, i in zip(
                    prompts, completions, answers, states, tasks, infos
                )
            ]
            rewards = await asyncio.gather(*coros)
            metric_name = task_completion_score.__name__
            return RolloutScores(
                reward=list(rewards), metrics={metric_name: list(rewards)}
            )

    TEST_CONCURRENCY = int(os.getenv("TB_TEST_CONCURRENCY", str(ROLLOUT_CONCURRENCY)))
    print(f"[TERMINALBENCH_ENV] Test concurrency: {TEST_CONCURRENCY}")
    rubric = ParallelTestRubric(parser=parser, max_parallel_tests=TEST_CONCURRENCY)

    # Create custom ToolEnv that sets up task context
    class TerminalBenchEnv(ToolEnv):
        def __init__(self, **kwargs):
            self.executor = executor
            tools = [execute_commands]
            super().__init__(tools=tools, max_turns=20, **kwargs)
            # Serialize only shared attributes; prefer thread-local context
            self._tool_call_lock = threading.Lock()

        def setup_state(self, state: dict, **kwargs):
            # Ensure execute_commands has the correct context early in the rollout
            info = state.get("info", {}) or {}
            execute_commands._current_task_id = info.get("task_id")
            execute_commands._current_task_path = info.get("task_path")
            execute_commands._current_info = info

            # Proactively create/start a terminal context so container is ready during first model turn
            try:
                task_id = info.get("task_id")
                task_path = info.get("task_path")
                if task_id and task_path:
                    # Favor speed for local runs unless explicitly overridden by user
                    os.environ.setdefault("TB_NO_REBUILD", "1")
                    os.environ.setdefault("TB_CLEANUP", "0")
                    self.executor.get_context(task_id, Path(task_path))
            except Exception as e:
                print(
                    f"[TERMINALBENCH_ENV]   Warning: failed to start context in setup_state: {e}"
                )

            return super().setup_state(state, **kwargs)

        def is_completed(self, messages, state, **kwargs):
            # Use default ToolEnv completion condition
            base_done = super().is_completed(messages, state, **kwargs)
            if not base_done:
                return False

            # Already evaluated and cleaned up for this rollout
            if isinstance(state, dict) and state.get("_tb_evaluated", False):
                return True

            # Run tests and cleanup now, cache results in state for rubric usage
            try:
                info = state.get("info", {}) or {}
                task_id = info.get("task_id")
                task_path_str = info.get("task_path")
                if not task_id or not task_path_str:
                    return True

                # Ensure a context exists (should already from setup_state)
                os.environ.setdefault("TB_NO_REBUILD", "1")
                os.environ.setdefault("TB_CLEANUP", "0")
                ctx = self.executor.get_context(task_id, Path(task_path_str))

                # Execute tests inside the container
                ran_ok, post_test_pane = ctx.run_tests(
                    timeout=float(info.get("max_test_timeout_sec", 180.0))
                )

                # Parse and determine success using Terminal-Bench's parser
                parsed = None
                success = False
                try:
                    parsed = ctx.trial_handler.parser.parse(post_test_pane)
                    all_passed = (
                        parsed is not None
                        and len(parsed) > 0
                        and all("PASSED" in str(v) for v in parsed.values())
                    )
                    success = bool(all_passed)
                except Exception as pe:
                    print(
                        f"[TERMINALBENCH_ENV] Parser error during is_completed eval: {pe}"
                    )
                    success = False

                # Capture agent commands log tail for debugging
                commands_log_tail = None
                try:
                    commands_log_path = ctx.trial_handler.trial_paths.commands_path
                    if commands_log_path.exists():
                        try:
                            log_text = commands_log_path.read_text(errors="replace")
                            log_lines = log_text.splitlines()
                            commands_log_tail = "\n".join(log_lines[-80:])
                        except Exception:
                            pass
                except Exception:
                    pass

                # Cache results in state for rubric
                state["_tb_evaluated"] = True
                state["terminalbench_ran_ok"] = bool(ran_ok)
                state["terminalbench_parsed_results"] = parsed
                state["terminalbench_parsed_success"] = bool(success)
                state["terminalbench_test_output"] = str(post_test_pane)
                if commands_log_tail is not None:
                    state["terminalbench_commands_log_tail"] = commands_log_tail

                # Cleanup container immediately after evaluation
                try:
                    print(
                        f"[TERMINALBENCH_ENV] 🧹 Cleaning up terminal for {task_id} at rollout end"
                    )
                    self.executor.cleanup_context(task_id)
                except Exception as ce:
                    print(
                        f"[TERMINALBENCH_ENV] Warning: cleanup failed for {task_id}: {ce}"
                    )

                return True
            except Exception as e:
                print(
                    f"[TERMINALBENCH_ENV] Warning: per-task eval in is_completed failed: {e}"
                )
                return True

        def _run_tool_call_threadsafe(
            self,
            tool_name: str,
            tool_args: dict,
            tool_call_id: str,
            info: dict,
            deadline: float | None,
        ) -> dict:
            # Thread-local context for parallel tool calls
            THREAD_LOCAL.task_id = info.get("task_id")
            THREAD_LOCAL.task_path = info.get("task_path")
            THREAD_LOCAL.info = info
            THREAD_LOCAL.deadline = deadline
            # Keep function attrs for legacy paths (minimal lock scope)
            with self._tool_call_lock:
                execute_commands._current_task_id = THREAD_LOCAL.task_id
                execute_commands._current_task_path = THREAD_LOCAL.task_path
                execute_commands._current_info = THREAD_LOCAL.info
            return self.call_tool(tool_name, tool_args, tool_call_id)

        async def rollout(
            self,
            client,
            model: str,
            prompt,
            answer: str = "",
            task: str = "default",
            info: dict | None = None,
            sampling_args=None,
            **kwargs,
        ) -> tuple[list[dict], dict]:
            """Async rollout that offloads blocking tool calls to threads for parallelism."""
            from copy import deepcopy as _dc

            info = info or {}
            # Establish rollout-wide time budget (align with TB harness semantics)
            env_total = os.getenv("TB_AGENT_TOTAL_TIMEOUT_SEC")
            try:
                total_budget = float(env_total) if env_total else None
            except Exception:
                total_budget = None
            if total_budget is None:
                try:
                    total_budget = float(info.get("max_agent_timeout_sec", 360.0))
                except Exception:
                    total_budget = 360.0
            start_time = time.time()
            deadline = start_time + float(total_budget)
            is_completed = False
            state = {
                "prompt": prompt,
                "completion": [],
                "answer": answer,
                "task": task,
                "info": info,
                "responses": [],
                "turn": 0,
            }
            state = self.setup_state(state)
            assert isinstance(prompt, list)
            completion: list[dict] = []
            rollout_msgs: list[dict] = _dc(prompt)
            while not is_completed:
                # Stop if rollout-wide budget exhausted
                if time.time() >= deadline:
                    print(
                        "[TERMINALBENCH_ENV] ⏳ Agent time budget exhausted; ending rollout."
                    )
                    break
                if self.is_completed(rollout_msgs, state, **kwargs):
                    is_completed = True
                    break
                response = await self.get_model_response(
                    client=client,
                    model=model,
                    prompt=rollout_msgs,
                    oai_tools=info.get("oai_tools", None),
                    sampling_args=sampling_args,
                    message_type=self.message_type,
                )
                state["responses"].append(response)
                # Assistant message
                response_text: str = response.choices[0].message.content or ""  # type: ignore
                response_message: dict = {
                    "role": "assistant",
                    "content": response_text,
                }
                if response.choices[0].message.tool_calls:
                    response_message["tool_calls"] = response.choices[
                        0
                    ].message.tool_calls  # type: ignore
                rollout_msgs.append(response_message)
                completion.append(response_message)
                state["turn"] += 1
                if (
                    self.is_completed(rollout_msgs, state, **kwargs)
                    or state["turn"] >= self.max_turns
                ):
                    is_completed = True
                else:
                    # Async tool execution: run each tool in a thread
                    assert "tool_calls" in rollout_msgs[-1]
                    tool_calls = rollout_msgs[-1]["tool_calls"] or []
                    tool_messages: list[dict] = []
                    # Execute sequentially to preserve order and avoid tmux conflicts per task
                    for tool_call in tool_calls:
                        tool_name: str = tool_call.function.name
                        import json as _json

                        tool_args: dict = _json.loads(tool_call.function.arguments)
                        tool_call_id: str = tool_call.id or ""
                        # Offload blocking execute_commands to a thread, with thread-safe context set
                        import asyncio as _asyncio

                        tool_message = await _asyncio.to_thread(
                            self._run_tool_call_threadsafe,
                            tool_name,
                            tool_args,
                            tool_call_id,
                            state.get("info", {}) or {},
                            deadline,
                        )
                        tool_messages.append(tool_message)
                    assert isinstance(rollout_msgs, list)
                    rollout_msgs += tool_messages
                    completion += tool_messages
            return completion, state

        async def a_generate(
            self,
            inputs,
            client=None,
            model: str | None = None,
            sampling_args=None,
            score_rollouts: bool = True,
            max_concurrent: int = -1,
            **kwargs,
        ):
            # Map external "max_concurrent_requests" to core "max_concurrent"
            mcr = kwargs.pop("max_concurrent_requests", None)
            if max_concurrent is None or max_concurrent < 1:
                if isinstance(mcr, int) and mcr > 0:
                    max_concurrent = mcr
                elif ROLLOUT_CONCURRENCY > 1:
                    max_concurrent = ROLLOUT_CONCURRENCY
                else:
                    # Leave as sequential
                    max_concurrent = -1
            print(
                f"[TERMINALBENCH_ENV] Rollout parallelism resolved to: {max_concurrent if max_concurrent > 0 else 'sequential'}"
            )
            return await super().a_generate(
                inputs,
                client=client,
                model=model,
                sampling_args=sampling_args,
                score_rollouts=score_rollouts,
                max_concurrent=max_concurrent,
                **kwargs,
            )

        def _init_state(self, state: dict):
            """Initialize the task context at the start of a rollout."""
            info = state.get("info", {})
            task_id = info.get("task_id")
            task_path = info.get("task_path")

            print("[TERMINALBENCH_ENV] 🚀 Initializing task state")
            print(f"[TERMINALBENCH_ENV]   Task ID: {task_id}")
            print(f"[TERMINALBENCH_ENV]   Task path available: {task_path is not None}")
            print(f"[TERMINALBENCH_ENV]   State keys: {list(state.keys())}")

            if task_id:
                execute_commands._current_task_id = task_id
                execute_commands._current_task_path = task_path
                execute_commands._current_info = info
                # Proactively create/start a terminal context so tests can run
                try:
                    if task_path:
                        self.executor.get_context(task_id, Path(task_path))
                except Exception as e:
                    print(
                        f"[TERMINALBENCH_ENV]   Warning: failed to start context early: {e}"
                    )
                print("[TERMINALBENCH_ENV]   ✅ Task context initialized")
            else:
                print("[TERMINALBENCH_ENV]   ❌ No task_id found in state")

        def env_response(self, messages, state, **kwargs):
            """Set up context for execute_commands function and delegate to parent"""
            info = state.get("info", {})
            task_id = info.get("task_id")
            task_path = info.get("task_path")

            print("[TERMINALBENCH_ENV] 🔧 Setting up task context")
            print(f"[TERMINALBENCH_ENV]   Task ID: {task_id}")
            print(f"[TERMINALBENCH_ENV]   Task path available: {task_path is not None}")
            print(f"[TERMINALBENCH_ENV]   State keys: {list(state.keys())}")
            print(
                f"[TERMINALBENCH_ENV]   Info keys: {list(info.keys()) if info else 'No info'}"
            )

            execute_commands._current_task_id = task_id
            execute_commands._current_task_path = task_path
            execute_commands._current_info = info

            print("[TERMINALBENCH_ENV]   Context set, delegating to parent ToolEnv")
            return super().env_response(messages, state, **kwargs)

    print(f"[TERMINALBENCH_ENV] Rollout concurrency: {ROLLOUT_CONCURRENCY}")

    env = TerminalBenchEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        message_type="chat",  # Required for function calling
    )

    # Attach executor and parallelism config to environment for cleanup and external control
    env._executor = executor  # type: ignore
    env.max_parallel_tasks = ROLLOUT_CONCURRENCY  # type: ignore

    # Removed custom __del__ method to prevent premature cleanup by garbage collector
    # Cleanup will be handled by atexit handlers and explicit cleanup in evaluation

    # Register additional cleanup for safety
    atexit.register(lambda: executor.cleanup() if executor else None)

    return env
