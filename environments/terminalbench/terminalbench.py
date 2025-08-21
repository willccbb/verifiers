"""
Terminal-Bench environment for Verifiers

This simplified environment reuses Terminal-Bench's native harness components
to run tasks in Docker via docker-compose and a tmux session, exposing a single
`execute_commands` tool for the agent. It avoids duplicating container logic.
"""

import atexit
import importlib
import importlib.util
import os
import types
import signal
import sys
import tempfile
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64

from datasets import Dataset

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv

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


class _TerminalContext:
    """Holds Terminal-Bench resources for a single task/session."""

    def __init__(self, task_path: Path, output_root: Path):
        # Create a TrialHandler to leverage naming, paths, and metadata
        trial_name = f"verifiers.1-of-1.{int(time.time())}"
        self.trial_handler = TrialHandler(
            trial_name=trial_name,
            input_path=task_path,
            output_path=output_root,
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
        self.terminal.start()
        # Run as configured user for agent session
        self.session = self.terminal.create_session(
            "agent", is_active_stream=False, as_configured_user=True
        )
        # Best-effort: ensure shell prompt is responsive before first use
        try:
            if self.session:
                self.session.send_keys(
                    ["echo __TB_SESSION_READY__", "Enter"],
                    block=True,
                    max_timeout_sec=10,
                )
                # Drain any incremental output produced during startup
                _ = self.session.get_incremental_output()
        except Exception:
            # Non-fatal; continue even if readiness probe fails
            pass

    def stop(self) -> None:
        try:
            self.terminal.stop()
        except Exception:
            pass

    def send_and_capture(self, command: str, timeout: float) -> Tuple[bool, str]:
        if not self.session:
            raise RuntimeError("Terminal session not started")

        # Execute command in a blocking manner and capture incremental output
        try:
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
                output = self.session.get_incremental_output()
            except Exception:
                output = ""
            return (
                False,
                output + f"\n[terminalbench] Command timed out after {timeout}s",
            )

        output = self.session.get_incremental_output()

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
        one_liner = (
            "tmpfile=$(mktemp /tmp/vf_cmd.XXXXXX.sh) && "
            f"echo '{encoded}' | base64 -d > \"$tmpfile\" && "
            'chmod +x "$tmpfile" && bash "$tmpfile"; status=$?; '
            'rm -f "$tmpfile"; echo ' + status_marker + ":$status"
        )

        ok, out = self.send_and_capture(one_liner, timeout)
        # Determine success from explicit status marker if present
        match = re.search(r"__VF_STATUS__:(\\d+)", out)
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

        # Get task context from the current conversation state
        task_id = execute_commands._current_task_id
        task_path_str = execute_commands._current_task_path

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

            # Determine timeout per command block
            info = getattr(execute_commands, "_current_info", {}) or {}
            # Priority: env var TB_CMD_TIMEOUT_SEC > info["max_agent_timeout_sec"] > default 180
            env_timeout = os.getenv("TB_CMD_TIMEOUT_SEC")
            try:
                timeout_sec = float(env_timeout) if env_timeout else None
            except Exception:
                timeout_sec = None
            if timeout_sec is None:
                try:
                    timeout_sec = float(info.get("max_agent_timeout_sec", 180))
                except Exception:
                    timeout_sec = 180.0

            # Execute as a single block to preserve heredocs and multi-line structure
            success, output = ctx.run_block(commands_str, timeout=timeout_sec)

            # Sanitize terminal output to avoid overly large or non-JSON-safe payloads
            def _strip_ansi(s: str) -> str:
                ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
                return ansi_escape.sub("", s)

            def _sanitize_output(s: str) -> str:
                s = _strip_ansi(s)
                # Replace non-printable characters except tab/newline/carriage return
                s = "".join(
                    ch if (ch.isprintable() or ch in "\t\n\r") else "�" for ch in s
                )
                return s

            output = _sanitize_output(output)

            # Truncate output if it's too long to prevent overwhelming the LLM
            # Keep conservative to reduce model request size and lower API error risk
            max_output_length = 8000
            if len(output) > max_output_length:
                truncated_output = (
                    output[:max_output_length]
                    + f"\n\n... [Output truncated. Total length: {len(output)} characters]"
                )
            else:
                truncated_output = output

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
                ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
                return ansi_escape.sub("", s)

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

    # Create rubric
    rubric = vf.Rubric(
        funcs=[task_completion_score],
        weights=[1.0],
        parser=parser,
        parallelize_scoring=False,
    )

    # Create custom ToolEnv that sets up task context
    class TerminalBenchEnv(ToolEnv):
        def __init__(self, **kwargs):
            self.executor = executor
            tools = [execute_commands]
            super().__init__(tools=tools, max_turns=20, **kwargs)

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

    env = TerminalBenchEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        message_type="chat",  # Required for function calling
    )

    # Attach executor to environment for cleanup
    env._executor = executor  # type: ignore

    # Removed custom __del__ method to prevent premature cleanup by garbage collector
    # Cleanup will be handled by atexit handlers and explicit cleanup in evaluation

    # Register additional cleanup for safety
    atexit.register(lambda: executor.cleanup() if executor else None)

    return env


def cleanup_all_docker_resources():
    """No-op shim retained for compatibility; Terminal manages cleanup itself."""
    pass
