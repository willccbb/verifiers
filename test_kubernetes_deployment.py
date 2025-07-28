#!/usr/bin/env python3
"""
Test script for Kubernetes deployment functionality
"""
import sys
import subprocess


def check_prerequisites():
    """Check if required tools are installed"""
    tools = {
        "kubectl": "kubectl version --client",
        "docker": "docker --version",
        "kind": "kind --version"
    }
    
    print("Checking prerequisites...")
    missing = []
    
    for tool, cmd in tools.items():
        try:
            subprocess.run(cmd.split(), capture_output=True, check=True)
            print(f"✓ {tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"✗ {tool} is not installed")
            missing.append(tool)
    
    if missing:
        print(f"\nPlease install missing tools: {', '.join(missing)}")
        return False
    return True


def test_imports():
    """Test that deployment module can be imported"""
    print("\nTesting imports...")
    try:
        # First check if yaml is available
        import yaml
        print("✓ yaml is available")
    except ImportError:
        print("✗ yaml is not installed. Run: uv add pyyaml")
        return False
        
    try:
        from verifiers.deployment import (
            KubernetesDeployment,
            LocalCluster, 
            RemoteCluster,
            DeploymentConfig
        )
        print("✓ Successfully imported deployment classes")
        return True
    except ImportError as e:
        print(f"✗ Failed to import deployment module: {e}")
        return False


def test_deployment_config():
    """Test creating a deployment configuration"""
    print("\nTesting deployment configuration...")
    try:
        from verifiers.deployment import DeploymentConfig
        
        config = DeploymentConfig(
            model="test-model",
            replicas=2,
            gpus_per_replica=0,
            namespace="test"
        )
        
        print(f"✓ Created deployment config: model={config.model}, replicas={config.replicas}")
        return True
    except Exception as e:
        print(f"✗ Failed to create deployment config: {e}")
        return False


def test_cluster_creation():
    """Test creating cluster objects (without actually creating clusters)"""
    print("\nTesting cluster object creation...")
    try:
        from verifiers.deployment import LocalCluster, RemoteCluster
        
        # Test local cluster
        local = LocalCluster(provider="kind", name="test-cluster")
        print(f"✓ Created LocalCluster: provider={local.provider}, name={local.name}")
        
        # Test remote cluster
        remote = RemoteCluster()
        print(f"✓ Created RemoteCluster: kubeconfig={remote.kubeconfig_path}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create cluster objects: {e}")
        return False


def main():
    """Run all tests"""
    print("=== Testing Kubernetes Deployment Module ===\n")
    
    tests = [
        ("Prerequisites", check_prerequisites),
        ("Module imports", test_imports),
        ("Deployment config", test_deployment_config),
        ("Cluster objects", test_cluster_creation),
    ]
    
    results = []
    for name, test_func in tests:
        if results and results[-1] is False:
            # Skip remaining tests if a critical test failed
            print(f"\nSkipping {name} due to previous failures")
            results.append(False)
        else:
            results.append(test_func())
    
    print("\n=== Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("\n✓ All tests passed! The Kubernetes deployment module is ready to use.")
        print("\nNext steps:")
        print("1. Run examples/deployment/deploy_local_k8s.py for local deployment")
        print("2. Run examples/deployment/deploy_remote_k8s.py for remote cluster deployment")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()