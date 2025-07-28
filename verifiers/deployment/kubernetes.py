"""
Kubernetes deployment for vLLM servers in verifiers.

Supports both local cluster creation (kind, minikube) and 
connection to existing remote clusters.
"""
import asyncio
import json
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


@dataclass
class DeploymentConfig:
    """Configuration for vLLM deployment on Kubernetes"""
    
    model: str
    replicas: int = 1
    gpus_per_replica: int = 1
    cpu_request: str = "4"
    memory_request: str = "16Gi"
    cpu_limit: str = "8"
    memory_limit: str = "32Gi"
    
    # vLLM specific settings
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    port: int = 8000
    enforce_eager: bool = True
    disable_log_requests: bool = True
    
    # Kubernetes settings
    namespace: str = "verifiers"
    service_type: str = "ClusterIP"  # or LoadBalancer, NodePort
    image: str = "vllm/vllm-openai:latest"
    
    # Additional vLLM args
    extra_vllm_args: Dict[str, str] = None
    
    def __post_init__(self):
        if self.extra_vllm_args is None:
            self.extra_vllm_args = {}


class KubernetesCluster(ABC):
    """Abstract base class for Kubernetes clusters"""
    
    @abstractmethod
    def setup(self) -> None:
        """Setup the cluster (create if local, configure if remote)"""
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Cleanup cluster resources"""
        pass
    
    @abstractmethod
    def get_kubeconfig(self) -> str:
        """Get the kubeconfig path for this cluster"""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if cluster is ready for deployments"""
        pass


class LocalCluster(KubernetesCluster):
    """Local Kubernetes cluster using kind or minikube"""
    
    def __init__(self, provider: str = "kind", name: str = "verifiers-cluster"):
        """
        Initialize local cluster
        
        Args:
            provider: "kind" or "minikube"
            name: Name of the cluster
        """
        self.provider = provider
        self.name = name
        self.kubeconfig_path = None
        
    def setup(self) -> None:
        """Create local cluster"""
        print(f"Setting up local {self.provider} cluster '{self.name}'...")
        
        if self.provider == "kind":
            self._setup_kind()
        elif self.provider == "minikube":
            self._setup_minikube()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
            
    def _setup_kind(self) -> None:
        """Setup kind cluster with GPU support if available"""
        # Check if cluster already exists
        result = subprocess.run(
            ["kind", "get", "clusters"], 
            capture_output=True, 
            text=True
        )
        
        if self.name in result.stdout.strip().split('\n'):
            print(f"Cluster '{self.name}' already exists")
        else:
            # Create kind cluster config
            config = {
                "kind": "Cluster",
                "apiVersion": "kind.x-k8s.io/v1alpha4",
                "nodes": [
                    {
                        "role": "control-plane",
                        "extraPortMappings": [
                            {
                                "containerPort": 30000,
                                "hostPort": 30000,
                                "protocol": "TCP"
                            }
                        ]
                    },
                    {
                        "role": "worker",
                        # Enable GPU passthrough if NVIDIA runtime is available
                        "extraMounts": [
                            {
                                "hostPath": "/dev/null",
                                "containerPath": "/var/run/nvidia-container-devices",
                                "readOnly": False
                            }
                        ] if self._check_nvidia_runtime() else []
                    }
                ]
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                config_path = f.name
            
            try:
                subprocess.run(
                    ["kind", "create", "cluster", "--name", self.name, "--config", config_path],
                    check=True
                )
            finally:
                os.unlink(config_path)
                
        # Get kubeconfig
        kubeconfig = subprocess.run(
            ["kind", "get", "kubeconfig", "--name", self.name],
            capture_output=True,
            text=True,
            check=True
        ).stdout
        
        # Save kubeconfig
        self.kubeconfig_path = os.path.expanduser(f"~/.kube/kind-{self.name}")
        with open(self.kubeconfig_path, 'w') as f:
            f.write(kubeconfig)
            
    def _setup_minikube(self) -> None:
        """Setup minikube cluster"""
        # Check if cluster exists
        result = subprocess.run(
            ["minikube", "status", "-p", self.name],
            capture_output=True
        )
        
        if result.returncode != 0:
            # Create cluster with GPU support if available
            cmd = ["minikube", "start", "-p", self.name, "--cpus=4", "--memory=8192"]
            
            if self._check_nvidia_runtime():
                cmd.extend(["--gpus", "all"])
                
            subprocess.run(cmd, check=True)
        else:
            print(f"Cluster '{self.name}' already exists")
            
        self.kubeconfig_path = os.path.expanduser(f"~/.kube/minikube-{self.name}")
        
        # Export kubeconfig
        subprocess.run(
            ["minikube", "kubectl", "-p", self.name, "config", "view", "--flatten"],
            stdout=open(self.kubeconfig_path, 'w'),
            check=True
        )
        
    def _check_nvidia_runtime(self) -> bool:
        """Check if NVIDIA runtime is available"""
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
            
    def teardown(self) -> None:
        """Delete local cluster"""
        print(f"Tearing down {self.provider} cluster '{self.name}'...")
        
        if self.provider == "kind":
            subprocess.run(["kind", "delete", "cluster", "--name", self.name])
        elif self.provider == "minikube":
            subprocess.run(["minikube", "delete", "-p", self.name])
            
        if self.kubeconfig_path and os.path.exists(self.kubeconfig_path):
            os.remove(self.kubeconfig_path)
            
    def get_kubeconfig(self) -> str:
        """Get kubeconfig path"""
        if not self.kubeconfig_path:
            raise RuntimeError("Cluster not set up yet")
        return self.kubeconfig_path
        
    def is_ready(self) -> bool:
        """Check if cluster is ready"""
        if not self.kubeconfig_path:
            return False
            
        result = subprocess.run(
            ["kubectl", "--kubeconfig", self.kubeconfig_path, "get", "nodes"],
            capture_output=True
        )
        return result.returncode == 0


class RemoteCluster(KubernetesCluster):
    """Connection to existing remote Kubernetes cluster"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None, context: Optional[str] = None):
        """
        Initialize remote cluster connection
        
        Args:
            kubeconfig_path: Path to kubeconfig file (uses default if None)
            context: Kubernetes context to use (uses current if None)
        """
        self.kubeconfig_path = kubeconfig_path or os.path.expanduser("~/.kube/config")
        self.context = context
        
    def setup(self) -> None:
        """Verify connection to remote cluster"""
        print("Connecting to remote cluster...")
        
        if not os.path.exists(self.kubeconfig_path):
            raise FileNotFoundError(f"Kubeconfig not found: {self.kubeconfig_path}")
            
        # Set context if specified
        if self.context:
            subprocess.run(
                ["kubectl", "--kubeconfig", self.kubeconfig_path, "config", "use-context", self.context],
                check=True
            )
            
        # Verify connection
        if not self.is_ready():
            raise RuntimeError("Cannot connect to Kubernetes cluster")
            
        print("Successfully connected to remote cluster")
        
    def teardown(self) -> None:
        """No teardown needed for remote clusters"""
        pass
        
    def get_kubeconfig(self) -> str:
        """Get kubeconfig path"""
        return self.kubeconfig_path
        
    def is_ready(self) -> bool:
        """Check if cluster is accessible"""
        cmd = ["kubectl", "--kubeconfig", self.kubeconfig_path, "get", "nodes"]
        if self.context:
            cmd.extend(["--context", self.context])
            
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0


class KubernetesDeployment:
    """Deploy vLLM servers on Kubernetes"""
    
    def __init__(self, cluster: KubernetesCluster, config: DeploymentConfig):
        """
        Initialize Kubernetes deployment
        
        Args:
            cluster: Kubernetes cluster instance
            config: Deployment configuration
        """
        self.cluster = cluster
        self.config = config
        self.deployment_name = f"vllm-{config.model.replace('/', '-').lower()}"
        self.service_name = f"{self.deployment_name}-service"
        
    def deploy(self) -> Dict[str, str]:
        """
        Deploy vLLM server to Kubernetes
        
        Returns:
            Dictionary with deployment info (service URL, etc.)
        """
        # Ensure cluster is ready
        self.cluster.setup()
        
        # Create namespace
        self._create_namespace()
        
        # Create deployment
        deployment_yaml = self._generate_deployment()
        self._apply_yaml(deployment_yaml)
        
        # Create service
        service_yaml = self._generate_service()
        self._apply_yaml(service_yaml)
        
        # Wait for deployment to be ready
        self._wait_for_deployment()
        
        # Get service endpoint
        endpoint = self._get_service_endpoint()
        
        return {
            "deployment_name": self.deployment_name,
            "service_name": self.service_name,
            "namespace": self.config.namespace,
            "endpoint": endpoint,
            "model": self.config.model
        }
        
    def _create_namespace(self) -> None:
        """Create namespace if it doesn't exist"""
        namespace_yaml = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.config.namespace
            }
        }
        
        # Check if namespace exists
        result = subprocess.run(
            [
                "kubectl", "--kubeconfig", self.cluster.get_kubeconfig(),
                "get", "namespace", self.config.namespace
            ],
            capture_output=True
        )
        
        if result.returncode != 0:
            self._apply_yaml(namespace_yaml)
            
    def _generate_deployment(self) -> dict:
        """Generate Kubernetes deployment YAML"""
        # Build vLLM command
        vllm_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model,
            "--port", str(self.config.port),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--data-parallel-size", str(self.config.data_parallel_size),
        ]
        
        if self.config.enforce_eager:
            vllm_cmd.append("--enforce-eager")
        if self.config.disable_log_requests:
            vllm_cmd.append("--disable-log-requests")
            
        # Add extra args
        for key, value in self.config.extra_vllm_args.items():
            vllm_cmd.extend([f"--{key}", str(value)])
            
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.deployment_name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": "vllm",
                    "model": self.config.model.replace("/", "-")
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": "vllm",
                        "deployment": self.deployment_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "vllm",
                            "deployment": self.deployment_name
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "vllm",
                                "image": self.config.image,
                                "command": vllm_cmd,
                                "ports": [
                                    {
                                        "containerPort": self.config.port,
                                        "name": "http"
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": self.config.cpu_request,
                                        "memory": self.config.memory_request,
                                    },
                                    "limits": {
                                        "cpu": self.config.cpu_limit,
                                        "memory": self.config.memory_limit,
                                    }
                                },
                                "env": [
                                    {
                                        "name": "OPENAI_API_KEY",
                                        "value": "EMPTY"
                                    },
                                    {
                                        "name": "VLLM_WORKER_MULTIPROC_METHOD",
                                        "value": "spawn"
                                    }
                                ],
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": self.config.port
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": self.config.port
                                    },
                                    "initialDelaySeconds": 60,
                                    "periodSeconds": 30
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        # Add GPU resources if requested
        if self.config.gpus_per_replica > 0:
            container = deployment["spec"]["template"]["spec"]["containers"][0]
            container["resources"]["limits"]["nvidia.com/gpu"] = str(self.config.gpus_per_replica)
            container["resources"]["requests"]["nvidia.com/gpu"] = str(self.config.gpus_per_replica)
            
        return deployment
        
    def _generate_service(self) -> dict:
        """Generate Kubernetes service YAML"""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.service_name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": "vllm",
                    "deployment": self.deployment_name
                }
            },
            "spec": {
                "type": self.config.service_type,
                "selector": {
                    "app": "vllm",
                    "deployment": self.deployment_name
                },
                "ports": [
                    {
                        "port": self.config.port,
                        "targetPort": self.config.port,
                        "name": "http"
                    }
                ]
            }
        }
        
        # Add NodePort if specified
        if self.config.service_type == "NodePort":
            service["spec"]["ports"][0]["nodePort"] = 30000
            
        return service
        
    def _apply_yaml(self, yaml_dict: dict) -> None:
        """Apply YAML to Kubernetes cluster"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_dict, f)
            yaml_path = f.name
            
        try:
            subprocess.run(
                [
                    "kubectl", "--kubeconfig", self.cluster.get_kubeconfig(),
                    "apply", "-f", yaml_path
                ],
                check=True
            )
        finally:
            os.unlink(yaml_path)
            
    def _wait_for_deployment(self, timeout: int = 300) -> None:
        """Wait for deployment to be ready"""
        print(f"Waiting for deployment {self.deployment_name} to be ready...")
        
        cmd = [
            "kubectl", "--kubeconfig", self.cluster.get_kubeconfig(),
            "-n", self.config.namespace,
            "wait", "--for=condition=available",
            f"deployment/{self.deployment_name}",
            f"--timeout={timeout}s"
        ]
        
        subprocess.run(cmd, check=True)
        print("Deployment is ready!")
        
    def _get_service_endpoint(self) -> str:
        """Get service endpoint URL"""
        # Get service info
        cmd = [
            "kubectl", "--kubeconfig", self.cluster.get_kubeconfig(),
            "-n", self.config.namespace,
            "get", "service", self.service_name,
            "-o", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        service_info = json.loads(result.stdout)
        
        if self.config.service_type == "ClusterIP":
            # For ClusterIP, return internal endpoint
            cluster_ip = service_info["spec"]["clusterIP"]
            return f"http://{cluster_ip}:{self.config.port}"
        elif self.config.service_type == "NodePort":
            # For NodePort, get node IP
            node_port = service_info["spec"]["ports"][0]["nodePort"]
            
            # Get first node IP
            cmd = [
                "kubectl", "--kubeconfig", self.cluster.get_kubeconfig(),
                "get", "nodes", "-o", 
                "jsonpath={.items[0].status.addresses[?(@.type=='InternalIP')].address}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            node_ip = result.stdout.strip()
            
            return f"http://{node_ip}:{node_port}"
        elif self.config.service_type == "LoadBalancer":
            # Wait for LoadBalancer IP
            ingress = service_info.get("status", {}).get("loadBalancer", {}).get("ingress", [])
            if ingress:
                lb_ip = ingress[0].get("ip") or ingress[0].get("hostname")
                return f"http://{lb_ip}:{self.config.port}"
            else:
                print("Warning: LoadBalancer IP not yet assigned")
                return f"http://pending:{self.config.port}"
                
    def delete(self) -> None:
        """Delete deployment and service"""
        print(f"Deleting deployment {self.deployment_name}...")
        
        # Delete deployment
        subprocess.run(
            [
                "kubectl", "--kubeconfig", self.cluster.get_kubeconfig(),
                "-n", self.config.namespace,
                "delete", "deployment", self.deployment_name
            ],
            capture_output=True
        )
        
        # Delete service  
        subprocess.run(
            [
                "kubectl", "--kubeconfig", self.cluster.get_kubeconfig(),
                "-n", self.config.namespace,
                "delete", "service", self.service_name
            ],
            capture_output=True
        )
        
    def scale(self, replicas: int) -> None:
        """Scale deployment to specified number of replicas"""
        print(f"Scaling deployment {self.deployment_name} to {replicas} replicas...")
        
        subprocess.run(
            [
                "kubectl", "--kubeconfig", self.cluster.get_kubeconfig(),
                "-n", self.config.namespace,
                "scale", f"deployment/{self.deployment_name}",
                f"--replicas={replicas}"
            ],
            check=True
        )
        
    def get_logs(self, tail: int = 100) -> str:
        """Get logs from deployment pods"""
        cmd = [
            "kubectl", "--kubeconfig", self.cluster.get_kubeconfig(),
            "-n", self.config.namespace,
            "logs", f"deployment/{self.deployment_name}",
            f"--tail={tail}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout


# Convenience function for port forwarding
def setup_port_forward(cluster: KubernetesCluster, namespace: str, service_name: str, 
                      local_port: int = 8000, remote_port: int = 8000) -> subprocess.Popen:
    """
    Setup kubectl port-forward for local access to service
    
    Returns:
        Popen object for the port-forward process
    """
    cmd = [
        "kubectl", "--kubeconfig", cluster.get_kubeconfig(),
        "-n", namespace,
        "port-forward", f"service/{service_name}",
        f"{local_port}:{remote_port}"
    ]
    
    process = subprocess.Popen(cmd)
    
    # Wait a bit for port forward to be established
    time.sleep(2)
    
    return process