# Using Kubernetes Backend for LiveCodeBench

LiveCodeBench now supports Kubernetes as an alternative to Docker for secure code execution. This enables better scalability and integration with cloud-native infrastructure.

## Installation

For Kubernetes support, install with the kubernetes extra:

```bash
pip install -e "environments/livecodebench[kubernetes]"
```

## Prerequisites

### For Local Development
- kubectl installed
- One of:
  - kind (recommended): `curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64 && chmod +x ./kind && sudo mv ./kind /usr/local/bin/`
  - minikube: `curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 && sudo install minikube-linux-amd64 /usr/local/bin/minikube`

### For Remote Clusters
- kubectl installed and configured
- Valid kubeconfig file
- Appropriate RBAC permissions to create pods and namespaces

## Usage

### Basic Usage with Local Cluster

```python
import verifiers as vf

# Use Kubernetes backend (creates local kind cluster automatically)
env = vf.load_environment(
    "livecodebench",
    backend="kubernetes",  # Switch to Kubernetes
    pool_size=30          # Number of pods to pre-allocate
)

# Use as normal
result = vf.evaluate(client, model, env)
```

### Using Remote Kubernetes Cluster

```python
import os

# Configure for remote cluster
os.environ["LIVECODEBENCH_K8S_LOCAL"] = "false"
os.environ["KUBECONFIG"] = "/path/to/kubeconfig"
os.environ["LIVECODEBENCH_K8S_CONTEXT"] = "production-cluster"

env = vf.load_environment("livecodebench", backend="kubernetes")
```

### Configuration via Environment Variables

- `LIVECODEBENCH_K8S_LOCAL`: "true" (default) or "false" - Use local or remote cluster
- `LIVECODEBENCH_K8S_PROVIDER`: "kind" (default) or "minikube" - Local cluster provider
- `LIVECODEBENCH_K8S_CONTEXT`: Kubernetes context to use (for remote clusters)
- `KUBECONFIG`: Path to kubeconfig file (defaults to ~/.kube/config)
- `DELETE_CLUSTER_ON_CLEANUP`: Set to any value to delete local cluster on cleanup

## Advantages of Kubernetes Backend

1. **Better Resource Management**: Kubernetes handles resource allocation and limits more effectively
2. **Scalability**: Easy to scale across multiple nodes
3. **Cloud Native**: Works seamlessly with cloud providers (EKS, GKE, AKS)
4. **Security**: Better isolation with network policies and security contexts
5. **Monitoring**: Integration with Kubernetes monitoring tools

## Architecture

The Kubernetes backend:
- Creates a dedicated namespace (`livecodebench`)
- Pre-allocates a pool of Python pods for fast execution
- Reuses pods between executions for performance
- Automatically cleans up resources on exit

## Troubleshooting

### Pod Creation Issues
```bash
# Check pod status
kubectl get pods -n livecodebench

# Check pod events
kubectl describe pod <pod-name> -n livecodebench
```

### Cleanup Issues
```bash
# Manual cleanup if needed
kubectl delete namespace livecodebench
```

### Performance Tuning

For better performance with large workloads:
```python
env = vf.load_environment(
    "livecodebench",
    backend="kubernetes",
    pool_size=50  # Increase pool size
)
```

## Comparison with Docker

| Feature | Docker | Kubernetes |
|---------|--------|------------|
| Setup Complexity | Low | Medium |
| Resource Limits | Process-level | Pod-level with guarantees |
| Scalability | Single node | Multi-node |
| Cloud Integration | Limited | Native |
| Overhead | Lower | Higher (but better for scale) |

Choose Docker for simple local development, Kubernetes for production or cloud deployments.