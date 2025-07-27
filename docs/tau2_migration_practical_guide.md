# Practical Migration Guide: τ²-bench to Verifiers

## Quick Start: What Can We Migrate Today?

Based on the analysis, here's what we can realistically migrate from τ²-bench to verifiers:

### ✅ **Immediately Feasible**
- Retail domain (agent-only tasks)
- Airline domain (agent-only tasks)  
- Basic tool implementations
- Policy documents
- Task definitions

### ❌ **Not Feasible Without Major Changes**
- Telecom domain (requires user tools)
- Dual-control scenarios
- User tool execution
- Full orchestration pattern

## Step-by-Step Migration Plan

### Step 1: Create Adapter for τ²-bench Tasks

```python
# environments/tau2_adapter/task_adapter.py

import json
from datasets import Dataset
from typing import List, Dict, Any
import verifiers as vf

class Tau2TaskAdapter:
    """Converts tau2-bench tasks to verifiers format."""
    
    def load_tau2_tasks(self, task_file: str) -> List[Dict[str, Any]]:
        """Load tasks from tau2 format."""
        with open(task_file, 'r') as f:
            return json.load(f)
    
    def convert_to_verifiers_dataset(self, tau2_tasks: List[Dict]) -> Dataset:
        """Convert tau2 tasks to verifiers dataset format."""
        dataset_rows = []
        
        for task in tau2_tasks:
            # Extract user instructions
            user_instructions = task.get("user_instructions", {})
            scenario = user_instructions.get("scenario", "")
            
            # Build prompt from initial messages if available
            initial_messages = []
            if task.get("initial_state", {}).get("message_history"):
                initial_messages = task["initial_state"]["message_history"]
            
            row = {
                "prompt": initial_messages or [{"role": "system", "content": "Customer service agent"}],
                "question": scenario,
                "info": {
                    "task_id": task["id"],
                    "expected_state": task.get("expected_state", {}),
                    "initial_state": task.get("initial_state", {}),
                    "user_instructions": user_instructions
                },
                "answer": "Task completed successfully",  # Placeholder
                "task": f"tau2_{task['domain']}"
            }
            dataset_rows.append(row)
        
        return Dataset.from_list(dataset_rows)
```

### Step 2: Create Environment Wrapper

```python
# environments/tau2_adapter/environment_wrapper.py

import verifiers as vf
from typing import List, Tuple, Dict, Any, Optional

class Tau2EnvironmentWrapper(vf.MultiTurnEnv):
    """Wraps tau2-bench environment for verifiers compatibility."""
    
    def __init__(self,
                 dataset: Dataset,
                 rubric: vf.Rubric,
                 tau2_env,  # Original tau2 environment
                 max_turns: int = 20,
                 **kwargs):
        super().__init__(dataset, rubric, **kwargs)
        self.tau2_env = tau2_env
        self.max_turns = max_turns
        self.turn_count = 0
        
    def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        """Check completion based on tau2 logic."""
        self.turn_count = sum(1 for m in messages if m["role"] == "assistant")
        
        # Max turns reached
        if self.turn_count >= self.max_turns:
            return True
            
        # Check for completion signals
        if messages:
            last_msg = messages[-1]
            if last_msg["role"] == "assistant":
                # Look for completion patterns
                content = last_msg.get("content", "").lower()
                if any(phrase in content for phrase in [
                    "task completed",
                    "anything else",
                    "have a great day"
                ]):
                    return True
                    
        return False
        
    def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> Tuple[vf.Messages, vf.State]:
        """Generate user response using simplified logic."""
        # Since we can't use tau2's orchestrator directly,
        # we'll implement a simplified user response
        
        # Get user instructions from state
        user_instructions = state.get("user_instructions", {})
        scenario = user_instructions.get("scenario", "")
        
        # Simple template-based responses
        # In real implementation, this would use an LLM
        user_response = self._generate_simple_user_response(
            messages, scenario, state
        )
        
        new_message = {"role": "user", "content": user_response}
        return [new_message], state
        
    def _generate_simple_user_response(self, messages, scenario, state):
        """Simplified user response generation."""
        # This is a placeholder - real implementation would be more sophisticated
        turn_responses = [
            "Yes, I need help with my order",
            "My order number is #W2378156",
            "I want to exchange the keyboard for one with clicky switches",
            "That sounds good, please proceed",
            "Thank you for your help!"
        ]
        
        turn = min(self.turn_count, len(turn_responses) - 1)
        return turn_responses[turn]
```

### Step 3: Port Tools with Adapter

```python
# environments/tau2_retail/tools_adapter.py

from verifiers import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any

class Tau2ToolAdapter(BaseTool):
    """Base adapter for tau2 tools."""
    
    def __init__(self, tau2_tool_func, name: str, description: str):
        self.tau2_tool_func = tau2_tool_func
        self.name = name
        self.description = description
        
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Execute tau2 tool and adapt response."""
        try:
            # Call original tau2 tool
            result = self.tau2_tool_func(**kwargs)
            
            # Adapt response format if needed
            if isinstance(result, str):
                return {"status": "success", "result": result}
            elif isinstance(result, dict):
                return result
            else:
                return {"status": "success", "data": str(result)}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Example: Adapting find_user_id_by_name_zip
class FindUserTool(Tau2ToolAdapter):
    """Find user by name and zip code."""
    
    class InputSchema(BaseModel):
        first_name: str = Field(description="First name")
        last_name: str = Field(description="Last name")
        zip_code: str = Field(description="Zip code")
    
    def __init__(self, tau2_tools):
        super().__init__(
            tau2_tool_func=tau2_tools.find_user_id_by_name_zip,
            name="find_user_id_by_name_zip",
            description="Find user ID by name and zip code"
        )
```

### Step 4: Create Simplified Rubric

```python
# environments/tau2_retail/rubric.py

import verifiers as vf
from typing import Dict, Any

def create_tau2_rubric(domain: str) -> vf.Rubric:
    """Create evaluation rubric for tau2 tasks."""
    
    def check_goal_achievement(completion, info, state, **kwargs) -> float:
        """Check if expected state was achieved."""
        expected_state = info.get("expected_state", {})
        
        # For retail: check if correct actions were taken
        if domain == "retail":
            # Check order modifications
            if "order_modifications" in expected_state:
                actual_mods = state.get("order_modifications", [])
                expected_mods = expected_state["order_modifications"]
                
                if len(actual_mods) == len(expected_mods):
                    return 1.0
                else:
                    return len(actual_mods) / max(len(expected_mods), 1)
                    
        return 0.0
        
    def check_efficiency(completion, info, state, **kwargs) -> float:
        """Check efficiency metrics."""
        tool_calls = state.get("tool_calls", [])
        expected_tools = info.get("expected_tools", 10)  # Assumed baseline
        
        if len(tool_calls) <= expected_tools:
            return 1.0
        else:
            # Penalize for excessive tool use
            return expected_tools / len(tool_calls)
            
    rubric = vf.Rubric(
        funcs=[check_goal_achievement, check_efficiency],
        weights=[0.8, 0.2]
    )
    
    return rubric
```

### Step 5: Complete Environment Loader

```python
# environments/tau2_retail/tau2_retail.py

import verifiers as vf
from pathlib import Path
import json

def load_environment(
    domain: str = "retail",
    num_train_examples: int = 50,
    num_eval_examples: int = 15,
    **kwargs
) -> vf.Environment:
    """Load tau2-bench environment in verifiers format."""
    
    # Load original tau2 components
    from tau2.domains.retail.environment import get_environment as get_tau2_env
    from tau2.domains.retail.environment import get_tasks
    
    # Get tau2 environment (without user tools)
    tau2_env = get_tau2_env(solo_mode=True)  # Important: solo_mode!
    
    # Load and convert tasks
    tau2_tasks = get_tasks()
    adapter = Tau2TaskAdapter()
    dataset = adapter.convert_to_verifiers_dataset(tau2_tasks)
    
    # Split dataset
    train_dataset = dataset.select(range(min(num_train_examples, len(dataset))))
    eval_dataset = dataset.select(range(
        num_train_examples, 
        min(num_train_examples + num_eval_examples, len(dataset))
    ))
    
    # Create adapted tools
    tools = []
    for tool_name, tool_func in tau2_env.tools.get_tools().items():
        adapted_tool = Tau2ToolAdapter(
            tau2_tool_func=tool_func,
            name=tool_name,
            description=f"Tool: {tool_name}"
        )
        tools.append(adapted_tool)
    
    # Create rubric
    rubric = create_tau2_rubric(domain)
    
    # Create wrapper environment
    env = Tau2EnvironmentWrapper(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        tau2_env=tau2_env,
        tools=tools,
        **kwargs
    )
    
    return env
```

## Testing the Migration

### 1. Basic Functionality Test

```python
# test_tau2_migration.py

import verifiers as vf
from environments.tau2_retail import load_environment

# Load environment
env = load_environment(domain="retail", num_train_examples=5)

# Test basic tool execution
tools = env.tools
find_user_tool = next(t for t in tools if t.name == "find_user_id_by_name_zip")

result = find_user_tool.invoke({
    "first_name": "John",
    "last_name": "Doe", 
    "zip_code": "12345"
})
print(f"Tool result: {result}")

# Test environment response
messages = [
    {"role": "assistant", "content": "Hello, how can I help you?"}
]
state = {"user_instructions": {"scenario": "Exchange product"}}

response_msgs, new_state = env.env_response(messages, state)
print(f"User response: {response_msgs[0]['content']}")
```

### 2. Run Evaluation

```python
# Run with existing verifiers infrastructure
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-mini")
agent_factory = vf.StandardAgentFactory(env, model)

# This should work with the adapted environment
results = env.evaluate(
    agent_factory=agent_factory,
    num_examples=5
)
```

## Limitations and Workarounds

### 1. **No User Tools**
- **Limitation**: Users can't execute tools
- **Workaround**: Pre-script user actions in scenarios

### 2. **Simplified Orchestration**
- **Limitation**: No complex multi-party orchestration
- **Workaround**: Sequential turn-taking only

### 3. **Limited State Management**
- **Limitation**: Single state dict instead of three
- **Workaround**: Namespace state keys (e.g., `user_`, `agent_`, `env_`)

### 4. **No Telecom Domain**
- **Limitation**: Requires dual-control
- **Workaround**: Focus on retail and airline domains

## Next Steps

1. **Implement basic adapter** (1-2 days)
2. **Port retail domain tools** (2-3 days)
3. **Test with GPT-4** (1 day)
4. **Compare results** with original tau2-bench
5. **Document differences** and limitations

## Success Criteria

- [ ] Can load tau2 tasks into verifiers
- [ ] Tools execute correctly
- [ ] Basic user simulation works
- [ ] Evaluation runs without errors
- [ ] Results are comparable (within 10% of original)