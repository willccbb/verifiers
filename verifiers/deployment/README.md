# Kubernetes Deployment for Verifiers

This module provides Kubernetes deployment capabilities for vLLM servers used in verifiers training and inference.

## Features

- **Local Cluster Support**: Automatically create and manage local Kubernetes clusters using kind or minikube
- **Remote Cluster Support**: Connect to existing Kubernetes clusters (EKS, GKE, AKS, on-prem, etc.)
- **GPU Support**: Automatic GPU resource allocation and scheduling
- **Auto-scaling**: Deploy multiple replicas for high-throughput inference
- **Custom vLLM Image**: Support for verifiers weight synchronization

## Installation

### Prerequisites

For local clusters:
```bash
# Option 1: kind (recommended for development)
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Option 2: minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# kubectl (required for both)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

For GPU support:
- NVIDIA GPU drivers
- NVIDIA Container Toolkit
- For Kubernetes: NVIDIA device plugin

### Python Dependencies

```bash
uv add pyyaml
```

## Quick Start

### Local Development

```python
from verifiers.deployment import KubernetesDeployment, LocalCluster, DeploymentConfig

# Create local cluster
cluster = LocalCluster(provider="kind", name="verifiers-dev")

# Configure deployment
config = DeploymentConfig(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    replicas=1,
    gpus_per_replica=0,  # Set to 1 if GPU available
    service_type="NodePort"
)

# Deploy
deployment = KubernetesDeployment(cluster, config)
info = deployment.deploy()

print(f"vLLM available at: {info['endpoint']}")
```

### Production Deployment

```python
from verifiers.deployment import KubernetesDeployment, RemoteCluster, DeploymentConfig

# Connect to existing cluster
cluster = RemoteCluster(
    kubeconfig_path="/path/to/kubeconfig",
    context="production-cluster"
)

# Production configuration
config = DeploymentConfig(
    model="meta-llama/Llama-2-70b-chat-hf",
    replicas=4,
    gpus_per_replica=8,
    tensor_parallel_size=8,
    service_type="LoadBalancer",
    namespace="ml-models",
    image="myregistry.com/verifiers/vllm-server:latest"
)

deployment = KubernetesDeployment(cluster, config)
info = deployment.deploy()
```

## Building Custom vLLM Image

The custom image includes verifiers weight synchronization support:

```bash
cd /path/to/verifiers
bash verifiers/deployment/build_image.sh

# Push to registry
docker tag verifiers/vllm-server:latest myregistry.com/verifiers/vllm-server:latest
docker push myregistry.com/verifiers/vllm-server:latest
```

## Integration with Training

```python
import verifiers as vf

# Deploy vLLM on Kubernetes
endpoint, deployment = deploy_vllm_server()

# Extract host and port
host = endpoint.split('//')[1].split(':')[0]
port = int(endpoint.split(':')[-1])

# Configure training
args = vf.grpo_defaults(
    vllm_server_host=host,
    vllm_server_port=port,
    max_concurrent=50  # Higher concurrency with K8s scaling
)

# Train as usual
trainer = vf.GRPOTrainer(model, tokenizer, env, args)
trainer.train()
```

## Configuration Options

### DeploymentConfig

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Model name/path | Required |
| `replicas` | Number of pod replicas | 1 |
| `gpus_per_replica` | GPUs per pod | 1 |
| `cpu_request` | CPU request | "4" |
| `memory_request` | Memory request | "16Gi" |
| `tensor_parallel_size` | Tensor parallelism | 1 |
| `data_parallel_size` | Data parallelism | 1 |
| `service_type` | ClusterIP/NodePort/LoadBalancer | "ClusterIP" |
| `namespace` | Kubernetes namespace | "verifiers" |
| `image` | Docker image | "vllm/vllm-openai:latest" |
| `extra_vllm_args` | Additional vLLM arguments | {} |

### Cluster Types

**LocalCluster**:
- `provider`: "kind" or "minikube"
- `name`: Cluster name

**RemoteCluster**:
- `kubeconfig_path`: Path to kubeconfig file
- `context`: Kubernetes context name

## Advanced Features

### Scaling

```python
# Scale deployment
deployment.scale(replicas=8)

# Get logs
logs = deployment.get_logs(tail=100)
print(logs)
```

### Port Forwarding

For local access to ClusterIP services:

```python
from verifiers.deployment.kubernetes import setup_port_forward

process = setup_port_forward(
    cluster, 
    namespace="verifiers",
    service_name="vllm-service",
    local_port=8000,
    remote_port=8000
)
```

### Monitoring

Monitor deployment health and performance:

```python
# Check deployment status
kubectl get pods -n verifiers

# View logs
kubectl logs -n verifiers deployment/vllm-qwen-qwen2.5-1.5b-instruct

# Check resource usage
kubectl top pods -n verifiers
```

## Troubleshooting

### Common Issues

1. **GPU not available**: Ensure NVIDIA device plugin is installed in the cluster
2. **Pod pending**: Check resource requests vs available cluster resources
3. **Connection refused**: Verify service type and endpoint configuration
4. **Image pull errors**: Ensure cluster has access to Docker registry

### Debug Commands

```bash
# Check pod status
kubectl get pods -n verifiers -o wide

# Describe pod for events
kubectl describe pod <pod-name> -n verifiers

# Check service endpoints
kubectl get svc -n verifiers

# Test connectivity
kubectl port-forward -n verifiers svc/vllm-service 8000:8000
curl http://localhost:8000/health
```

## Best Practices

1. **Resource Allocation**: Set appropriate resource requests/limits based on model size
2. **Replicas**: Use multiple replicas for production workloads
3. **Namespaces**: Isolate deployments using Kubernetes namespaces
4. **Monitoring**: Set up Prometheus/Grafana for production monitoring
5. **Security**: Use private registries and RBAC for production deployments

## Examples

See the `examples/deployment/` directory for complete examples:
- `deploy_local_k8s.py`: Local development setup
- `deploy_remote_k8s.py`: Production deployment
- `train_with_k8s.py`: Training integration