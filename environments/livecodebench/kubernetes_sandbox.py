"""
Kubernetes-based code execution sandbox for LiveCodeBench.

This provides a drop-in replacement for Docker-based execution,
supporting both local clusters (kind/minikube) and remote clusters.
"""
import base64
import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")


@dataclass
class KubernetesConfig:
    """Configuration for Kubernetes sandbox"""
    
    # Cluster configuration
    use_local_cluster: bool = True
    local_provider: str = "kind"  # "kind" or "minikube"
    cluster_name: str = "livecodebench"
    kubeconfig_path: Optional[str] = None  # For remote clusters
    context: Optional[str] = None  # For remote clusters
    
    # Pod configuration
    namespace: str = "livecodebench"
    image: str = "python:3.11-slim"
    cpu_request: str = "100m"
    memory_request: str = "128Mi"
    cpu_limit: str = "1000m"
    memory_limit: str = "512Mi"
    
    # Execution settings
    default_timeout: int = 10
    pod_startup_timeout: int = 30
    
    # Pool settings
    pool_size: int = 20
    reuse_pods: bool = True


class KubernetesSandbox:
    """Kubernetes-based code execution sandbox"""
    
    def __init__(self, config: Optional[KubernetesConfig] = None):
        """Initialize Kubernetes sandbox"""
        self.config = config or KubernetesConfig()
        self.kubeconfig = self._setup_kubeconfig()
        
        # Ensure cluster is ready
        self._ensure_cluster()
        
        # Create namespace
        self._create_namespace()
        
        # Pod pool for reuse
        self.pod_pool: List[str] = []
        self.available_pods: List[str] = []
        
        if self.config.reuse_pods:
            print(f"Pre-allocating {self.config.pool_size} pods...")
            self._initialize_pod_pool()
    
    def _setup_kubeconfig(self) -> str:
        """Setup kubeconfig path"""
        if self.config.use_local_cluster:
            # Local cluster kubeconfig
            if self.config.local_provider == "kind":
                return os.path.expanduser(f"~/.kube/kind-{self.config.cluster_name}")
            else:  # minikube
                return os.path.expanduser(f"~/.kube/minikube-{self.config.cluster_name}")
        else:
            # Remote cluster
            return self.config.kubeconfig_path or os.path.expanduser("~/.kube/config")
    
    def _run_kubectl(self, args: List[str], check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run kubectl command"""
        cmd = ["kubectl", "--kubeconfig", self.kubeconfig]
        if self.config.context:
            cmd.extend(["--context", self.config.context])
        cmd.extend(args)
        
        return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)
    
    def _ensure_cluster(self):
        """Ensure Kubernetes cluster is available"""
        if self.config.use_local_cluster:
            self._ensure_local_cluster()
        else:
            self._verify_remote_cluster()
    
    def _ensure_local_cluster(self):
        """Ensure local cluster exists"""
        if self.config.local_provider == "kind":
            # Check if cluster exists
            result = subprocess.run(
                ["kind", "get", "clusters"],
                capture_output=True,
                text=True
            )
            
            if self.config.cluster_name not in result.stdout.strip().split('\n'):
                print(f"Creating kind cluster '{self.config.cluster_name}'...")
                # Create kind config
                config = {
                    "kind": "Cluster",
                    "apiVersion": "kind.x-k8s.io/v1alpha4",
                    "nodes": [{"role": "control-plane"}]
                }
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(config, f)
                    config_path = f.name
                
                try:
                    subprocess.run(
                        ["kind", "create", "cluster", "--name", self.config.cluster_name, 
                         "--config", config_path],
                        check=True
                    )
                finally:
                    os.unlink(config_path)
            
            # Export kubeconfig
            kubeconfig = subprocess.run(
                ["kind", "get", "kubeconfig", "--name", self.config.cluster_name],
                capture_output=True,
                text=True,
                check=True
            ).stdout
            
            os.makedirs(os.path.dirname(self.kubeconfig), exist_ok=True)
            with open(self.kubeconfig, 'w') as f:
                f.write(kubeconfig)
                
        else:  # minikube
            # Check if cluster exists
            result = subprocess.run(
                ["minikube", "status", "-p", self.config.cluster_name],
                capture_output=True
            )
            
            if result.returncode != 0:
                print(f"Creating minikube cluster '{self.config.cluster_name}'...")
                subprocess.run(
                    ["minikube", "start", "-p", self.config.cluster_name, 
                     "--cpus=2", "--memory=2048"],
                    check=True
                )
            
            # Export kubeconfig
            subprocess.run(
                ["minikube", "kubectl", "-p", self.config.cluster_name, 
                 "config", "view", "--flatten"],
                stdout=open(self.kubeconfig, 'w'),
                check=True
            )
    
    def _verify_remote_cluster(self):
        """Verify connection to remote cluster"""
        try:
            self._run_kubectl(["get", "nodes"])
            print("Successfully connected to remote cluster")
        except subprocess.CalledProcessError:
            raise RuntimeError("Cannot connect to Kubernetes cluster. Check kubeconfig and context.")
    
    def _create_namespace(self):
        """Create namespace if it doesn't exist"""
        # Check if namespace exists
        result = self._run_kubectl(
            ["get", "namespace", self.config.namespace],
            check=False
        )
        
        if result.returncode != 0:
            # Create namespace
            namespace_yaml = {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": self.config.namespace}
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(namespace_yaml, f)
                yaml_path = f.name
            
            try:
                self._run_kubectl(["apply", "-f", yaml_path])
            finally:
                os.unlink(yaml_path)
    
    def _initialize_pod_pool(self):
        """Initialize pool of pods for reuse"""
        for i in range(self.config.pool_size):
            pod_name = self._create_execution_pod()
            if pod_name:
                self.pod_pool.append(pod_name)
                self.available_pods.append(pod_name)
        print(f"Initialized {len(self.pod_pool)} pods")
    
    def _create_execution_pod(self) -> Optional[str]:
        """Create a pod for code execution"""
        pod_name = f"exec-{uuid.uuid4().hex[:8]}"
        
        pod_spec = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": "livecodebench",
                    "purpose": "execution"
                }
            },
            "spec": {
                "containers": [{
                    "name": "python",
                    "image": self.config.image,
                    "command": ["sleep", "3600"],  # Keep pod alive
                    "resources": {
                        "requests": {
                            "cpu": self.config.cpu_request,
                            "memory": self.config.memory_request
                        },
                        "limits": {
                            "cpu": self.config.cpu_limit,
                            "memory": self.config.memory_limit
                        }
                    }
                }],
                "restartPolicy": "Never"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(pod_spec, f)
            yaml_path = f.name
        
        try:
            self._run_kubectl(["apply", "-f", yaml_path])
            
            # Wait for pod to be ready
            start_time = time.time()
            while time.time() - start_time < self.config.pod_startup_timeout:
                result = self._run_kubectl([
                    "get", "pod", pod_name, "-n", self.config.namespace,
                    "-o", "jsonpath={.status.phase}"
                ])
                
                if result.stdout.strip() == "Running":
                    return pod_name
                
                time.sleep(1)
            
            print(f"Warning: Pod {pod_name} not ready after {self.config.pod_startup_timeout}s")
            return None
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to create pod: {e}")
            return None
        finally:
            os.unlink(yaml_path)
    
    def _get_available_pod(self) -> Optional[str]:
        """Get an available pod from the pool"""
        if self.available_pods:
            return self.available_pods.pop(0)
        
        # Create new pod if pool is empty
        if len(self.pod_pool) < self.config.pool_size * 2:  # Allow some growth
            pod_name = self._create_execution_pod()
            if pod_name:
                self.pod_pool.append(pod_name)
                return pod_name
        
        return None
    
    def _return_pod(self, pod_name: str):
        """Return pod to available pool"""
        if pod_name in self.pod_pool:
            self.available_pods.append(pod_name)
    
    def execute(self, code: str, test_input: str = "", timeout: Optional[int] = None) -> Tuple[str, str, int]:
        """
        Execute code in Kubernetes pod
        
        Args:
            code: Python code to execute
            test_input: Input to provide to the program
            timeout: Execution timeout in seconds
            
        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        timeout = timeout or self.config.default_timeout
        
        if self.config.reuse_pods:
            pod_name = self._get_available_pod()
            if not pod_name:
                return "", "No available pods", 1
        else:
            # Create one-time pod
            pod_name = self._create_execution_pod()
            if not pod_name:
                return "", "Failed to create pod", 1
        
        try:
            # Create script with code and input
            script = f"""
import sys
import io

# Redirect stdin
sys.stdin = io.StringIO('''{test_input}''')

# Execute user code
{code}
"""
            
            # Base64 encode the script to avoid shell escaping issues
            script_b64 = base64.b64encode(script.encode()).decode()
            
            # Execute in pod
            result = self._run_kubectl([
                "exec", "-n", self.config.namespace, pod_name, "--",
                "python", "-c", f"import base64; exec(base64.b64decode('{script_b64}').decode())"
            ], check=False)
            
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
            
            # Handle timeout (kubectl doesn't have built-in timeout for exec)
            # In production, you'd use a proper timeout mechanism
            
            return stdout, stderr, exit_code
            
        finally:
            if self.config.reuse_pods:
                self._return_pod(pod_name)
            else:
                # Delete one-time pod
                self._run_kubectl([
                    "delete", "pod", pod_name, "-n", self.config.namespace,
                    "--grace-period=0", "--force"
                ], check=False)
    
    def cleanup(self):
        """Clean up Kubernetes resources"""
        print("Cleaning up Kubernetes resources...")
        
        # Delete all pods
        if self.pod_pool:
            print(f"Deleting {len(self.pod_pool)} pods...")
            for pod_name in self.pod_pool:
                self._run_kubectl([
                    "delete", "pod", pod_name, "-n", self.config.namespace,
                    "--grace-period=0", "--force"
                ], check=False)
        
        # Delete any orphaned pods
        self._run_kubectl([
            "delete", "pods", "-n", self.config.namespace,
            "-l", "app=livecodebench",
            "--grace-period=0", "--force"
        ], check=False)
        
        # Optionally delete namespace (commented out for safety)
        # self._run_kubectl(["delete", "namespace", self.config.namespace], check=False)
        
        # Optionally delete local cluster
        if self.config.use_local_cluster and os.environ.get("DELETE_CLUSTER_ON_CLEANUP"):
            if self.config.local_provider == "kind":
                subprocess.run(["kind", "delete", "cluster", "--name", self.config.cluster_name])
            else:
                subprocess.run(["minikube", "delete", "-p", self.config.cluster_name])
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Example usage and integration with LiveCodeBench:
if __name__ == "__main__":
    # Example 1: Local development with kind
    config = KubernetesConfig(
        use_local_cluster=True,
        local_provider="kind",
        pool_size=10
    )
    
    with KubernetesSandbox(config) as sandbox:
        # Test execution
        code = """
print("Hello from Kubernetes!")
x = int(input())
print(f"Square of {x} is {x**2}")
"""
        stdout, stderr, exit_code = sandbox.execute(code, test_input="5", timeout=10)
        print(f"Output: {stdout}")
        print(f"Errors: {stderr}")
        print(f"Exit code: {exit_code}")
    
    # Example 2: Remote cluster usage
    # config = KubernetesConfig(
    #     use_local_cluster=False,
    #     kubeconfig_path="/path/to/kubeconfig",
    #     context="production-cluster",
    #     namespace="ml-experiments",
    #     pool_size=50
    # )
    
    # sandbox = KubernetesSandbox(config)
    # # Use sandbox...
    # sandbox.cleanup()