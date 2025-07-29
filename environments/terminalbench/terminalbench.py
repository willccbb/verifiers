"""
Terminal-Bench Dataset Implementation for Verifiers

This implementation uses the Terminal-Bench dataset from HuggingFace (ia03/terminal-bench)
with Docker-based execution for real terminal environment tasks.
"""

import atexit
import hashlib
import io
import signal
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import docker  # type: ignore
from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv


class TerminalContainer:
    """Manages a single Docker container for terminal task execution"""

    def __init__(self, docker_client, task_data: Dict[str, Any]):
        self.docker_client = docker_client
        self.task_data = task_data
        self.container = None
        self.task_dir = None
        self._setup_container()

    def _setup_container(self):
        """Set up the Docker container for this task"""
        # Extract task to temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="terminalbench_")
        self.task_dir = Path(self.temp_dir) / self.task_data["task_id"]
        self.task_dir.mkdir(parents=True, exist_ok=True)

        # Verify archive integrity
        archive_bytes = self.task_data["archive"]
        expected_hash = self.task_data["tar_sha256"]
        actual_hash = hashlib.sha256(archive_bytes).hexdigest()

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Archive integrity check failed for {self.task_data['task_id']}"
            )

        # Extract archive
        try:
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
                tar.extractall(self.task_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to extract archive: {e}")

        # Build and start container
        image_name = f"terminalbench_{self.task_data['task_id']}".lower().replace(
            "-", "_"
        )

        try:
            # Build image
            self.docker_client.images.build(
                path=str(self.task_dir),
                tag=image_name,
                rm=True,
                labels={"terminalbench": "true"},
            )

            # Start container
            self.container = self.docker_client.containers.run(
                image=image_name,
                command=["sleep", "infinity"],
                detach=True,
                remove=False,
                labels={"terminalbench": "true"},
                mem_limit="2g",
                memswap_limit="2g",
                cpu_quota=100000,  # 1 CPU
                pids_limit=100,
                environment={
                    "TEST_DIR": "/tests",
                    "T_BENCH_TEST_DIR": "/tests",
                    "T_BENCH_CONTAINER_LOGS_PATH": "/var/log/tbench",
                },
            )

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to setup container: {e}")

    def execute_commands(self, commands: str, timeout: int = 180) -> Tuple[bool, str]:
        """Execute commands in the container and return success, output"""
        if not self.container:
            return False, "Container not available"

        try:
            result = self.container.exec_run(
                cmd=["bash", "-c", commands],
                stdout=True,
                stderr=True,
                tty=False,
                user="root",
            )

            output = result.output.decode("utf-8", errors="replace")
            success = result.exit_code == 0

            return success, output

        except Exception as e:
            return False, f"Command execution failed: {e}"

    def run_tests(self, timeout: int = 30) -> Tuple[bool, str]:
        """Run the final tests and return success, output"""
        if not self.container:
            return False, "Container not available"

        print("\n🧪 RUNNING TERMINALBENCH TESTS FOR TASK 🧪")
        print(f"Task ID: {self.task_data.get('task_id', 'unknown')}")
        print(f"Test timeout: {timeout} seconds")

        try:
            # Copy tests to container if they exist
            tests_dir = self.task_dir / "tests"
            if tests_dir.exists():
                print(f"📁 Found tests directory: {tests_dir}")
                print("📁 Test files:")
                for test_file in tests_dir.iterdir():
                    print(f"   - {test_file.name}")

                # Create tests directory in container
                self.container.exec_run(["mkdir", "-p", "/tests"])

                # Copy test files
                tar_buffer = io.BytesIO()
                with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                    tar.add(tests_dir, arcname="tests")
                tar_buffer.seek(0)

                self.container.put_archive("/", tar_buffer)
                print("✅ Tests copied to container")
            else:
                print(f"❌ No tests directory found at {tests_dir}")

            # Determine test command
            test_script = self.task_dir / "run-tests.sh"
            if test_script.exists():
                print(f"🔧 Found custom test script: {test_script}")
                # Copy custom test script
                tar_buffer = io.BytesIO()
                with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
                    tar.add(test_script, arcname="run-tests.sh")
                tar_buffer.seek(0)
                self.container.put_archive("/", tar_buffer)
                test_cmd = "bash /run-tests.sh"
                print(f"🔧 Using custom test command: {test_cmd}")
            else:
                print("🔧 No custom test script found, using default pytest")
                # Use default pytest runner
                test_cmd = """
                cd /tests
                echo "=== PYTEST TEST EXECUTION ==="
                echo "Working directory: $(pwd)"
                echo "Files in test directory:"
                ls -la
                echo "=== INSTALLING PYTEST ==="
                python -m pip install pytest > /dev/null 2>&1 || pip install pytest > /dev/null 2>&1
                echo "=== RUNNING PYTEST ==="
                python -m pytest test_outputs.py -v -s --tb=short
                """
                print("🔧 Using default pytest command")

            print("\n🚀 EXECUTING TESTS...")
            # Execute tests
            result = self.container.exec_run(
                cmd=["bash", "-c", test_cmd],
                stdout=True,
                stderr=True,
                tty=False,
                user="root",
            )

            output = result.output.decode("utf-8", errors="replace")
            success = result.exit_code == 0

            print("\n📊 TEST RESULTS:")
            print(f"Exit code: {result.exit_code}")
            print(f"Success: {success}")
            print(f"Output length: {len(output)} characters")

            # Parse and display test results in detail
            if "FAILED" in output or "ERROR" in output:
                print("\n❌ DETAILED FAILURE ANALYSIS:")
                lines = output.split("\n")
                for i, line in enumerate(lines):
                    if "FAILED" in line or "ERROR" in line or "AssertionError" in line:
                        print(f"   Line {i + 1}: {line}")
                        # Print context around failures
                        for j in range(max(0, i - 2), min(len(lines), i + 3)):
                            if j != i:
                                print(f"   Context {j + 1}: {lines[j]}")

            if "PASSED" in output:
                print("\n✅ PASSED TESTS:")
                lines = output.split("\n")
                for line in lines:
                    if "PASSED" in line:
                        print(f"   {line}")

            # Show full output for debugging
            print("\n📝 FULL TEST OUTPUT:")
            print("=" * 50)
            print(output)
            print("=" * 50)

            return success, output

        except Exception as e:
            error_msg = f"Test execution failed: {e}"
            print(f"❌ {error_msg}")
            return False, error_msg

    def cleanup(self):
        """Clean up container and temporary files"""
        if self.container:
            try:
                self.container.stop(timeout=5)
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

        if hasattr(self, "temp_dir") and self.temp_dir:
            try:
                import shutil

                shutil.rmtree(self.temp_dir)
            except Exception:
                pass

    # Removed __del__ method to prevent premature cleanup by garbage collector


class TerminalTaskExecutor:
    """Manages Docker containers for terminal tasks"""

    def __init__(self):
        """Initialize Docker-based terminal task executor"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()

            # Clean up any orphaned containers from previous runs
            self._cleanup_orphaned_containers()

            # Track active containers
            self.active_containers = {}
            self._register_cleanup_handlers()

            print("Terminal task executor with Docker initialized successfully")

        except Exception as e:
            raise RuntimeError(
                f"Docker is required but not available: {e}\n"
                "Please ensure Docker is installed and running:\n"
                "  - Install Docker: https://docs.docker.com/get-docker/\n"
                "  - Start Docker daemon: sudo systemctl start docker\n"
                "  - Add user to docker group: sudo usermod -aG docker $USER"
            )

    def _cleanup_orphaned_containers(self):
        """Clean up any containers from previous runs"""
        try:
            containers = self.docker_client.containers.list(
                all=True, filters={"label": "terminalbench=true"}
            )

            if containers:
                print(
                    f"Cleaning up {len(containers)} orphaned containers from previous runs..."
                )
                for container in containers:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: Failed to clean up orphaned containers: {e}")

    def _register_cleanup_handlers(self):
        """Register cleanup handlers for various exit scenarios"""
        atexit.register(self.cleanup)

        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, cleaning up...")
            self.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def get_container(
        self, task_id: str, task_data: Dict[str, Any]
    ) -> TerminalContainer:
        """Get or create a container for the given task"""
        if task_id not in self.active_containers:
            self.active_containers[task_id] = TerminalContainer(
                self.docker_client, task_data
            )
        return self.active_containers[task_id]

    def cleanup_container(self, task_id: str):
        """Clean up a specific container"""
        if task_id in self.active_containers:
            self.active_containers[task_id].cleanup()
            del self.active_containers[task_id]

    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up Docker resources...")

        # Clean up all active containers
        for task_id in list(self.active_containers.keys()):
            self.cleanup_container(task_id)

        try:
            self.docker_client.containers.prune(filters={"label": "terminalbench=true"})
        except Exception as e:
            print(f"Warning: Failed to prune Docker resources: {e}")

    # Removed __del__ method to prevent premature cleanup by garbage collector

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        return False


def load_terminalbench_dataset(
    dataset_name: str = "ia03/terminal-bench",
    split: str = "test",  # Terminal-bench now uses "test" split
    num_examples: int = -1,
) -> Dataset:
    """Load the Terminal-Bench dataset from HuggingFace"""

    print(f"Loading Terminal-Bench dataset: {dataset_name}")
    print(f"Split: {split}")

    try:
        # Try different loading approaches based on the dataset structure
        dataset = None

        # Load dataset with explicit data files mapping
        try:
            print("Attempting to load dataset with explicit data files mapping...")

            # Map the test files to the requested split explicitly
            if split == "test":
                data_files = {split: "data/test-*.parquet"}
            else:
                # Fallback for other splits (though only test is expected)
                data_files = {split: "data/*.parquet"}

            full_dataset = load_dataset(dataset_name, data_files=data_files)

            # Check available splits
            if isinstance(full_dataset, dict):
                available_splits = list(full_dataset.keys())
                print(f"Available splits: {available_splits}")

                # Use the requested split if available
                if split in full_dataset:
                    dataset = full_dataset[split]
                    print(f"✅ Dataset loaded successfully: {len(dataset)} tasks")
                else:
                    print(f"Split '{split}' not found in {available_splits}")
                    raise Exception(f"Split '{split}' not found in dataset")
            else:
                # Single split dataset (shouldn't happen with explicit mapping)
                dataset = full_dataset
                print(f"✅ Dataset loaded successfully: {len(dataset)} tasks")

        except Exception as e1:
            print(f"Dataset loading failed: {e1}")
            raise Exception(
                "Failed to load Terminal-Bench dataset. Please check if the dataset exists and is accessible."
            )

        if dataset is None:
            raise Exception("Dataset loading failed - dataset is None")

        # Limit examples if specified
        if num_examples > 0 and len(dataset) > num_examples:  # type: ignore
            dataset = dataset.select(range(num_examples))  # type: ignore
            print(f"Limited to {num_examples} examples")

        return dataset  # type: ignore

    except Exception as e:
        print(f"Error loading Terminal-Bench dataset: {e}")
        print("\nTroubleshooting tips:")
        print(
            "1. Check if the dataset exists at: https://huggingface.co/datasets/ia03/terminal-bench"
        )
        print("2. Verify internet connection")
        print("3. Try updating the datasets library: pip install -U datasets")
        print("4. Check if the dataset is private and requires authentication")
        print(
            "5. The dataset files may be in a data/ subdirectory - this is handled automatically"
        )
        raise


def load_environment(
    dataset_name: str = "ia03/terminal-bench",
    split: str = "test",  # Terminal-bench uses "test" split
    num_examples: int = -1,
) -> vf.ToolEnv:
    """Load Terminal-Bench environment with proper multi-turn support

    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to use
        num_examples: Number of examples to load (-1 for all)

    Returns:
        ToolEnv: Configured environment with proper multi-turn interaction
    """
    # Load dataset
    dataset = load_terminalbench_dataset(dataset_name, split, num_examples)

    # Initialize task executor
    executor = TerminalTaskExecutor()

    # Convert dataset to verifiers format
    converted_dataset = []
    for idx, example in enumerate(dataset):
        task_id = example.get("task_id", f"task_{idx}")  # type: ignore
        base_description = example.get("base_description", "")  # type: ignore
        difficulty = example.get("difficulty", "unknown")  # type: ignore
        tags = example.get("tags", [])  # type: ignore
        category = example.get("category", "")  # type: ignore

        # Format the prompt (Terminal-Bench style)
        prompt = "You are an AI assistant helping to complete terminal tasks. You have access to a Linux terminal environment.\n\n"
        prompt += f"## Task: {task_id}\n\n"
        prompt += f"**Description:** {base_description}\n\n"
        if category:
            prompt += f"**Category:** {category}\n"
        if difficulty:
            prompt += f"**Difficulty:** {difficulty}\n"
        if tags:
            prompt += f"**Tags:** {', '.join(tags)}\n"
        prompt += "\n**CRITICAL INSTRUCTIONS:**\n"
        prompt += (
            "- You MUST use the `execute_commands` function to run shell commands\n"
        )
        prompt += "- ALWAYS provide brief reasoning AND immediately follow with execute_commands\n"
        prompt += "- NEVER make statements without immediate action\n"
        prompt += "- BAD: 'Now I need to create the file.' [END]\n"
        prompt += "- GOOD: 'Now I need to create the file.' [IMMEDIATE tool_call]\n"
        prompt += "- Continue working until the task is completely finished\n"
        prompt += "- Never stop until all requirements are implemented and tested\n\n"
        prompt += "**Your approach:**\n"
        prompt += "1. State what you're doing next (1 sentence)\n"
        prompt += "2. IMMEDIATELY call execute_commands - no exceptions\n"
        prompt += "3. Based on results, state next step and immediately execute\n"
        prompt += "4. Repeat continuously until the entire task is complete\n\n"
        prompt += "**Important notes:**\n"
        prompt += "- You are working in a Docker container with full root access\n"
        prompt += "- Commands will be executed sequentially in bash\n"
        prompt += "- Avoid interactive commands that require user input\n"
        prompt += "- If you need to install software, use package managers like apt, pip, etc.\n"
        prompt += (
            "- Your goal is to complete the task such that the test suite will pass\n"
        )
        prompt += "- You can make multiple function calls to iterate and check your progress\n"
        prompt += (
            "- REMEMBER: Always respond with execute_commands calls, never just text\n"
        )
        prompt += "- When the task is completely done, say 'TASK COMPLETED' in your final message"

        converted_dataset.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "answer": "",  # Will be filled by the model
                "info": {
                    "task_id": task_id,
                    "task_data": example,
                    "base_description": base_description,
                    "difficulty": difficulty,
                    "tags": tags,
                    "category": category,
                    "max_agent_timeout_sec": example.get("max_agent_timeout_sec", 180),  # type: ignore
                    "max_test_timeout_sec": example.get("max_test_timeout_sec", 30),  # type: ignore
                },
            }
        )

    # Create dataset
    dataset = Dataset.from_list(converted_dataset)

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

        # Get task data from the current conversation state
        # This is set up by the environment during conversation
        task_id = execute_commands._current_task_id
        task_data = execute_commands._current_task_data

        print(f"[TERMINALBENCH]   Current task_id: {task_id}")
        print(f"[TERMINALBENCH]   Current task_data available: {task_data is not None}")

        if not task_id or not task_data:
            return "❌ ERROR: Terminal environment not properly initialized."

        try:
            # Handle both string and array inputs for commands
            if isinstance(commands, str):
                commands_str = commands
            elif isinstance(commands, list):
                commands_str = "\n".join(str(cmd) for cmd in commands)
            else:
                return f"❌ ERROR: Commands must be a string or array of strings, got {type(commands)}"

            # Get or create container for this task
            container = executor.get_container(task_id, task_data)

            # Execute commands
            success, output = container.execute_commands(commands_str, timeout=180)

            # Truncate output if it's too long to prevent overwhelming the LLM
            max_output_length = (
                8000  # Increased from 2000 to 8000 for better test result visibility
            )
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
    execute_commands._current_task_data = None

    # Define rubric functions for evaluation
    def task_completion_score(completion, info, parser, state) -> float:
        """Evaluate task completion by running the final tests"""
        print("\n⚖️  EVALUATING TASK COMPLETION ⚖️")

        try:
            task_id = info["task_id"]
            task_data = info["task_data"]

            print(f"Task ID: {task_id}")
            print(f"Task data available: {task_data is not None}")

            # Get the container (should exist if agent ran any commands)
            if task_id not in executor.active_containers:
                print(f"❌ No active container found for task {task_id}")
                print(f"Active containers: {list(executor.active_containers.keys())}")
                return 0.0  # No commands were executed

            container = executor.active_containers[task_id]
            print(f"✅ Found active container for task {task_id}")

            # Run the final tests
            print("🔬 Running Terminal-Bench test suite...")
            success, output = container.run_tests(timeout=info["max_test_timeout_sec"])

            print("\n📋 FINAL EVALUATION RESULT:")
            print(f"Tests passed: {success}")
            print(f"Score: {1.0 if success else 0.0}")

            if not success:
                print("❌ Task failed Terminal-Bench tests")
            else:
                print("✅ Task passed all Terminal-Bench tests!")

            # Clean up container after testing
            print(f"🧹 Cleaning up container for {task_id}")
            executor.cleanup_container(task_id)

            return 1.0 if success else 0.0

        except Exception as e:
            print(f"❌ Error during task evaluation: {e}")
            print(f"Exception type: {type(e)}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

            # Clean up container even if evaluation failed
            try:
                if task_id in executor.active_containers:
                    print(f"🧹 Cleaning up container for {task_id} after error")
                    executor.cleanup_container(task_id)
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
            task_data = info.get("task_data")

            print("[TERMINALBENCH_ENV] 🚀 Initializing task state")
            print(f"[TERMINALBENCH_ENV]   Task ID: {task_id}")
            print(f"[TERMINALBENCH_ENV]   Task data available: {task_data is not None}")
            print(f"[TERMINALBENCH_ENV]   State keys: {list(state.keys())}")

            if task_id:
                execute_commands._current_task_id = task_id
                execute_commands._current_task_data = task_data
                print("[TERMINALBENCH_ENV]   ✅ Task context initialized")
            else:
                print("[TERMINALBENCH_ENV]   ❌ No task_id found in state")

        def env_response(self, messages, state, **kwargs):
            """Set up context for execute_commands function and delegate to parent"""
            info = state.get("info", {})
            task_id = info.get("task_id")
            task_data = info.get("task_data")

            print("[TERMINALBENCH_ENV] 🔧 Setting up task context")
            print(f"[TERMINALBENCH_ENV]   Task ID: {task_id}")
            print(f"[TERMINALBENCH_ENV]   Task data available: {task_data is not None}")
            print(f"[TERMINALBENCH_ENV]   State keys: {list(state.keys())}")
            print(
                f"[TERMINALBENCH_ENV]   Info keys: {list(info.keys()) if info else 'No info'}"
            )

            execute_commands._current_task_id = task_id
            execute_commands._current_task_data = task_data

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
    """Utility function to clean up ALL Docker resources related to Terminal-Bench

    This is useful for manual cleanup if resources were left behind.
    """
    try:
        client = docker.from_env()

        # Remove all containers with our label
        containers = client.containers.list(
            all=True, filters={"label": "terminalbench=true"}
        )
        print(f"Found {len(containers)} Terminal-Bench containers to clean up...")

        for container in containers:
            try:
                container.stop(timeout=2)
                container.remove(force=True)
                print(f"Removed container {container.short_id}")
            except Exception as e:
                print(f"Failed to remove container {container.short_id}: {e}")

        # Remove all images with terminalbench prefix
        images = client.images.list(filters={"label": "terminalbench=true"})
        print(f"Found {len(images)} Terminal-Bench images to clean up...")

        for image in images:
            try:
                client.images.remove(image.id, force=True)
                print(f"Removed image {image.short_id}")
            except Exception as e:
                print(f"Failed to remove image {image.short_id}: {e}")

        # Prune system resources
        print("Pruning Docker system resources...")
        client.containers.prune()
        client.images.prune()

        print("Docker cleanup complete!")

    except Exception as e:
        print(f"Error during Docker cleanup: {e}")
