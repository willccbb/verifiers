"""
Example: Deploy vLLM server on remote/existing Kubernetes cluster
"""
import os
from verifiers.deployment import KubernetesDeployment, RemoteCluster, DeploymentConfig


def main():
    # Connect to existing cluster
    # Option 1: Use default kubeconfig (~/.kube/config)
    cluster = RemoteCluster()
    
    # Option 2: Specify custom kubeconfig and context
    # cluster = RemoteCluster(
    #     kubeconfig_path="/path/to/kubeconfig",
    #     context="my-cluster-context"
    # )
    
    # Configure deployment for production use
    config = DeploymentConfig(
        model="meta-llama/Llama-2-7b-chat-hf",  # Or your preferred model
        replicas=2,  # Multiple replicas for HA
        gpus_per_replica=1,  # GPU per replica
        cpu_request="8",
        memory_request="32Gi",
        cpu_limit="16",
        memory_limit="64Gi",
        tensor_parallel_size=1,
        data_parallel_size=1,
        service_type="LoadBalancer",  # Or ClusterIP for internal use
        namespace="ml-models",  # Custom namespace
        # Use custom image with verifiers support
        # image="myregistry.com/verifiers/vllm-server:latest",
        extra_vllm_args={
            "max-model-len": "4096",
            "gpu-memory-utilization": "0.9",
            "max-num-seqs": "256",
        }
    )
    
    # Create deployment manager
    deployment = KubernetesDeployment(cluster, config)
    
    try:
        # Deploy vLLM server
        print("\n=== Deploying vLLM server to remote cluster ===")
        info = deployment.deploy()
        
        print(f"\nDeployment successful!")
        print(f"  Deployment: {info['deployment_name']}")
        print(f"  Service: {info['service_name']}")
        print(f"  Namespace: {info['namespace']}")
        print(f"  Endpoint: {info['endpoint']}")
        
        # Show deployment status
        print("\n=== Checking deployment status ===")
        logs = deployment.get_logs(tail=30)
        print(logs)
        
        # Scale deployment if needed
        print("\n=== Scaling options ===")
        user_input = input("Scale deployment? [y/N]: ")
        if user_input.lower() == 'y':
            new_replicas = int(input("Number of replicas: "))
            deployment.scale(new_replicas)
            print(f"Scaled to {new_replicas} replicas")
        
        print("\n=== Deployment ready! ===")
        print(f"vLLM endpoint: {info['endpoint']}")
        print(f"\nTo use in your training scripts, set:")
        print(f"  export VLLM_ENDPOINT='{info['endpoint']}'")
        print(f"\nOr in Python:")
        print(f"  vllm_base_url = '{info['endpoint']}/v1'")
        
        # Example: Using with OpenAI client
        print("\n=== Example usage ===")
        print("""
from openai import OpenAI

client = OpenAI(
    base_url=f"{info['endpoint']}/v1",
    api_key="EMPTY"
)

response = client.completions.create(
    model=config.model,
    prompt="Hello, world!",
    max_tokens=100
)
print(response.choices[0].text)
        """)
        
    except Exception as e:
        print(f"\nError during deployment: {e}")
        
        # Check cluster connectivity
        if not cluster.is_ready():
            print("\nCannot connect to Kubernetes cluster!")
            print("Please check:")
            print("  1. kubeconfig is valid")
            print("  2. kubectl can connect to the cluster")
            print("  3. You have necessary permissions")
        raise
    finally:
        # Cleanup options
        print("\n=== Cleanup ===")
        user_input = input("Delete deployment? [y/N]: ")
        if user_input.lower() == 'y':
            deployment.delete()
            print("Deployment deleted")


if __name__ == "__main__":
    main()