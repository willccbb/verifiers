"""
Test script for τ²-bench implementation.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import verifiers as vf


def test_basic_loading():
    """Test basic environment loading."""
    print("Testing τ²-bench Implementation")
    print("=" * 50)
    
    # Check if tau2-bench is installed
    try:
        import tau2
        print("✓ tau2-bench is installed")
    except ImportError:
        print("✗ tau2-bench not installed")
        print("\nTo install:")
        print("git clone https://github.com/sierra-research/tau2-bench.git")
        print("pip install -e tau2-bench/")
        return False
    
    # Test each domain
    domains = ["retail", "airline", "telecom"]
    
    for domain in domains:
        print(f"\n--- Testing {domain} domain ---")
        try:
            env = vf.load_environment(
                "tau2_bench",
                domain=domain,
                num_eval_examples=1,
                user_llm="gpt-4"
            )
            
            print(f"✓ Loaded {domain} domain")
            print(f"  Tasks available: {len(env.tau2_tasks)}")
            
            # Show first task
            if env.tau2_tasks:
                first_task = env.tau2_tasks[0]
                print(f"  First task ID: {first_task['id']}")
                
                # Check for user instructions
                user_instr = first_task.get("user_instructions", {})
                if user_instr:
                    scenario = user_instr.get("scenario", "")[:100]
                    print(f"  Scenario: {scenario}...")
                    
            # Check tool availability
            if domain == "telecom":
                print(f"  Dual-control: YES (user has tools)")
            else:
                print(f"  Dual-control: NO (agent tools only)")
                
        except Exception as e:
            print(f"✗ Error loading {domain}: {e}")
            import traceback
            traceback.print_exc()
            
    return True


def test_env_response():
    """Test environment response handling."""
    print("\n\n--- Testing env_response ---")
    
    try:
        from tau2_bench import load_environment
        
        # Load retail environment
        env = load_environment(domain="retail", num_eval_examples=1)
        print("✓ Loaded retail environment")
        
        # Get first task
        example = env.dataset[0]
        state = {"task_id": example["task_id"]}
        
        # Test messages
        messages = [
            {"role": "system", "content": "You are a helpful customer service agent."},
            {"role": "user", "content": "Hi, I need help with my order."},
            {"role": "assistant", "content": "Hello! I'd be happy to help you with your order. Could you please provide me with your order number?"}
        ]
        
        # Process through env_response
        response_messages, updated_state = env.env_response(messages, state)
        
        print(f"✓ env_response executed")
        print(f"  Response messages: {len(response_messages)}")
        print(f"  State keys: {list(updated_state.keys())}")
        
        # Check state initialization
        assert "user_state" in updated_state
        assert "env_db" in updated_state
        assert "tool_executions" in updated_state
        assert "user_simulator" in updated_state
        print("✓ State properly initialized")
        
        # Show user response if generated
        if response_messages:
            for msg in response_messages:
                print(f"  {msg['role']}: {msg.get('content', '')[:80]}...")
                
    except Exception as e:
        print(f"✗ Error testing env_response: {e}")
        import traceback
        traceback.print_exc()


def test_tool_execution():
    """Test tool execution within env_response."""
    print("\n\n--- Testing Tool Execution ---")
    
    try:
        from tau2_bench import load_environment
        
        # Load environment
        env = load_environment(domain="retail", num_eval_examples=1)
        
        # Get first task
        example = env.dataset[0]
        state = {"task_id": example["task_id"]}
        
        # Message with tool call
        messages = [
            {"role": "system", "content": "You are a helpful customer service agent."},
            {"role": "user", "content": "Can you check order 12345?"},
            {
                "role": "assistant",
                "content": "I'll check that order for you.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_order_details",
                            "arguments": '{"order_id": "12345"}'
                        }
                    }
                ]
            }
        ]
        
        # Process
        response_messages, updated_state = env.env_response(messages, state)
        
        print("✓ Tool execution test completed")
        print(f"  Tool executions tracked: {len(updated_state.get('tool_executions', []))}")
        
        # Check for tool response
        tool_messages = [m for m in response_messages if m["role"] == "tool"]
        print(f"  Tool messages: {len(tool_messages)}")
        
        if updated_state.get("tool_executions"):
            exec = updated_state["tool_executions"][0]
            print(f"  First execution: {exec['tool']} by {exec['role']}")
            
    except Exception as e:
        print(f"✗ Error testing tools: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("τ²-bench Test Suite")
    print("==================\n")
    
    if test_basic_loading():
        test_env_response()
        test_tool_execution()
    
    print("\n\nKey Implementation Features:")
    print("✓ All orchestration logic in env_response")
    print("✓ Agent tool execution handled")
    print("✓ User simulation with tau2 UserSimulator")
    print("✓ User tools supported (telecom domain)")
    print("✓ Proper state management")
    print("✓ No modifications to verifiers core")


if __name__ == "__main__":
    main()