#!/usr/bin/env python3
"""Simple test runner to verify tests pass."""

import subprocess
import sys

def run_tests():
    """Run the test suite."""
    print("Running verifiers test suite...")
    print("-" * 60)
    
    try:
        # Run pytest
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd="/workspace",
            env={"PYTHONPATH": "/workspace"},
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())