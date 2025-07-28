"""
Example: Training with vLLM deployed on Kubernetes
"""
import os
import verifiers as vf
from verifiers.deployment import KubernetesDeployment, RemoteCluster, DeploymentConfig


def deploy_vllm_server():
    """Deploy vLLM server on Kubernetes and return endpoint"""
    
    # Connect to cluster (adjust as needed)
    cluster = RemoteCluster()
    
    # Configure deployment
    config = DeploymentConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        replicas=3,  # Multiple replicas for parallel inference
        gpus_per_replica=1,
        cpu_request="8",
        memory_request="32Gi",
        cpu_limit="16", 
        memory_limit="64Gi",
        tensor_parallel_size=1,
        data_parallel_size=1,  # Data parallelism handled by replicas
        service_type="ClusterIP",  # Internal cluster access
        namespace="verifiers-training",
        extra_vllm_args={
            "max-model-len": "2048",
            "gpu-memory-utilization": "0.9",
        }
    )
    
    deployment = KubernetesDeployment(cluster, config)
    info = deployment.deploy()
    
    return info['endpoint'], deployment


def main():
    # Deploy vLLM server
    print("Deploying vLLM server on Kubernetes...")
    endpoint, deployment = deploy_vllm_server()
    
    # Extract host and port from endpoint
    # endpoint format: http://host:port
    host = endpoint.split('//')[1].split(':')[0]
    port = int(endpoint.split(':')[-1])
    
    print(f"vLLM server deployed at {host}:{port}")
    
    try:
        # Load environment
        env = vf.load_environment("gsm8k")
        
        # Load model for training
        model, tokenizer = vf.get_model_and_tokenizer(
            "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller model for training
        )
        
        # Configure training with Kubernetes vLLM endpoint
        args = vf.grpo_defaults(
            run_name="k8s-training-demo",
            num_train_epochs=1,
            logging_steps=10,
            
            # Point to Kubernetes vLLM server
            vllm_server_host=host,
            vllm_server_port=port,
            
            # Increase concurrency for multiple replicas
            max_concurrent=50,  # Can handle more with K8s scaling
            
            # Other training parameters
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            num_generations=16,
            learning_rate=1e-6,
        )
        
        # Create trainer
        trainer = vf.GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            env=env,
            args=args,
        )
        
        print("Starting training with Kubernetes-deployed vLLM...")
        trainer.train()
        
        print("Training completed!")
        
    finally:
        # Cleanup
        print("\nCleaning up Kubernetes deployment...")
        user_input = input("Delete vLLM deployment? [y/N]: ")
        if user_input.lower() == 'y':
            deployment.delete()
            print("Deployment deleted")


def advanced_example():
    """Advanced example with auto-scaling and monitoring"""
    
    from verifiers.deployment.kubernetes import setup_port_forward
    import threading
    import requests
    
    # Deploy with auto-scaling configuration
    cluster = RemoteCluster()
    
    config = DeploymentConfig(
        model="meta-llama/Llama-2-13b-chat-hf",
        replicas=2,  # Start with 2, scale based on load
        gpus_per_replica=2,  # 2 GPUs for tensor parallelism
        tensor_parallel_size=2,
        data_parallel_size=1,
        service_type="ClusterIP",
        namespace="verifiers-training",
        # Custom image with verifiers support
        image="myregistry.com/verifiers/vllm-server:latest",
    )
    
    deployment = KubernetesDeployment(cluster, config)
    info = deployment.deploy()
    
    # Set up monitoring in a separate thread
    def monitor_deployment():
        while True:
            try:
                # Get current replica count
                logs = deployment.get_logs(tail=10)
                # Parse logs for request rate, queue size, etc.
                
                # Auto-scale based on metrics
                # This is a simplified example - in production use HPA
                # (Horizontal Pod Autoscaler) with custom metrics
                
                time.sleep(30)
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    monitor_thread = threading.Thread(target=monitor_deployment, daemon=True)
    monitor_thread.start()
    
    # Continue with training...
    print("Advanced deployment with monitoring active")
    
    return deployment


if __name__ == "__main__":
    main()
    
    # Uncomment for advanced example
    # deployment = advanced_example()