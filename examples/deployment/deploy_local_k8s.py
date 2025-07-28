"""
Example: Deploy vLLM server on local Kubernetes cluster
"""
import time
from verifiers.deployment import KubernetesDeployment, LocalCluster, DeploymentConfig
from verifiers.deployment.kubernetes import setup_port_forward


def main():
    # Create local cluster (kind or minikube)
    cluster = LocalCluster(provider="kind", name="verifiers-demo")
    
    # Configure deployment
    config = DeploymentConfig(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        replicas=1,
        gpus_per_replica=0,  # Set to 1 if you have GPU support
        cpu_request="2",
        memory_request="8Gi",
        cpu_limit="4", 
        memory_limit="16Gi",
        tensor_parallel_size=1,
        data_parallel_size=1,
        service_type="NodePort",  # For local access
        # Use custom image if built, otherwise default vLLM image
        # image="verifiers/vllm-server:latest", 
    )
    
    # Create deployment manager
    deployment = KubernetesDeployment(cluster, config)
    
    try:
        # Deploy vLLM server
        print("\n=== Deploying vLLM server to Kubernetes ===")
        info = deployment.deploy()
        
        print(f"\nDeployment successful!")
        print(f"  Deployment: {info['deployment_name']}")
        print(f"  Service: {info['service_name']}")
        print(f"  Namespace: {info['namespace']}")
        print(f"  Endpoint: {info['endpoint']}")
        
        # For local clusters, also set up port forwarding for easy access
        if config.service_type == "ClusterIP":
            print("\nSetting up port forward for local access...")
            pf_process = setup_port_forward(
                cluster, 
                config.namespace, 
                deployment.service_name,
                local_port=8000,
                remote_port=config.port
            )
            print("vLLM server accessible at http://localhost:8000")
        
        # Show logs
        print("\n=== Recent logs ===")
        logs = deployment.get_logs(tail=20)
        print(logs)
        
        # Test the deployment
        print("\n=== Testing deployment ===")
        import requests
        
        # Use NodePort endpoint or localhost if port-forwarded
        test_url = info['endpoint'] if config.service_type == "NodePort" else "http://localhost:8000"
        
        try:
            response = requests.get(f"{test_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Health check passed!")
            else:
                print(f"✗ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Could not connect to server: {e}")
            
        print("\n=== Deployment ready for use! ===")
        print(f"You can now use the vLLM endpoint in your training scripts:")
        print(f"  vllm_server_host = '{test_url.split('//')[1].split(':')[0]}'")
        print(f"  vllm_server_port = {test_url.split(':')[-1]}")
        
        # Keep running to maintain port forward
        if config.service_type == "ClusterIP":
            print("\nPress Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                pf_process.terminate()
                
    except Exception as e:
        print(f"\nError during deployment: {e}")
        raise
    finally:
        # Cleanup
        print("\nCleaning up...")
        user_input = input("Delete deployment? [y/N]: ")
        if user_input.lower() == 'y':
            deployment.delete()
            
        user_input = input("Delete cluster? [y/N]: ")
        if user_input.lower() == 'y':
            cluster.teardown()


if __name__ == "__main__":
    main()