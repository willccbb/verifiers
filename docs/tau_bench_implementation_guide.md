# τ-bench Implementation Guide for Verifiers

## Overview

This guide provides a step-by-step approach to migrate τ-bench (Tool-Agent-User benchmark) to the verifiers infrastructure. τ-bench tests agent capabilities in realistic customer service scenarios with dynamic user interactions, tool use, and policy compliance.

## Key Components Mapping

### 1. Environment Structure

**τ-bench Components** → **Verifiers Equivalents**
- `Env` class → `MultiTurnEnv`
- `Tool` classes → `BaseTool` implementations
- `User` simulator → Custom environment response logic
- `Task` → Dataset entries with expected states

### 2. Core Implementation Plan

```python
# environments/tau_bench_retail/tau_bench_retail.py

import verifiers as vf
from typing import List, Tuple, Dict, Any, Optional
from datasets import Dataset
import json
from pathlib import Path

class TauBenchRetailEnv(vf.MultiTurnEnv):
    """τ-bench retail environment implementation."""
    
    def __init__(self, 
                 dataset: Dataset,
                 rubric: vf.Rubric,
                 max_turns: int = 20,
                 user_model: str = "gpt-4.1-mini",
                 **kwargs):
        super().__init__(dataset, rubric, **kwargs)
        self.max_turns = max_turns
        self.user_model = user_model
        self.db_state = {}  # Simulated database
        self.policies = self._load_policies()
        
    def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        """Check if task is completed or max turns reached."""
        # Count turns
        turn_count = sum(1 for m in messages if m["role"] == "assistant")
        if turn_count >= self.max_turns:
            return True
            
        # Check if agent has indicated task completion
        last_message = messages[-1] if messages else None
        if last_message and last_message["role"] == "assistant":
            if "task completed" in last_message["content"].lower():
                return True
                
        return False
        
    def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> Tuple[vf.Messages, vf.State]:
        """Simulate user response based on scenario."""
        # Get last assistant message
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant_msg = msg
                break
                
        # Generate user response based on scenario
        user_response = self._generate_user_response(
            messages=messages,
            state=state,
            last_assistant_msg=last_assistant_msg
        )
        
        # Update state with user response
        state["user_responses"] = state.get("user_responses", [])
        state["user_responses"].append(user_response)
        
        # Return new message and updated state
        new_message = {"role": "user", "content": user_response}
        return [new_message], state
```

### 3. Tools Implementation

```python
# environments/tau_bench_retail/tools.py

from verifiers import BaseTool
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class FindUserByNameZipTool(BaseTool):
    """Find user ID by name and zip code."""
    
    name = "find_user_id_by_name_zip"
    description = "Find a user's ID by their first name, last name, and zip code"
    
    class InputSchema(BaseModel):
        first_name: str = Field(description="User's first name")
        last_name: str = Field(description="User's last name")
        zip: str = Field(description="User's zip code")
    
    def _run(self, first_name: str, last_name: str, zip: str) -> Dict[str, Any]:
        # Simulate database lookup
        # In real implementation, this would query actual data
        mock_users = {
            ("Yusuf", "Rossi", "19122"): "yusuf_rossi_9620",
            # Add more mock data
        }
        
        user_id = mock_users.get((first_name, last_name, zip))
        if user_id:
            return {"status": "success", "user_id": user_id}
        else:
            return {"status": "not_found", "message": "No user found with provided details"}

class GetOrderDetailsTool(BaseTool):
    """Get order details by order ID."""
    
    name = "get_order_details"
    description = "Retrieve details of an order by its ID"
    
    class InputSchema(BaseModel):
        order_id: str = Field(description="Order ID (e.g., #W2378156)")
    
    def _run(self, order_id: str) -> Dict[str, Any]:
        # Mock order data
        orders = {
            "#W2378156": {
                "status": "delivered",
                "items": [
                    {"item_id": "1151293680", "product_id": "1656367028", "name": "Mechanical Keyboard"},
                    {"item_id": "4983901480", "product_id": "4896585277", "name": "Smart Thermostat"}
                ],
                "payment_method": "credit_card_9513926"
            }
        }
        
        order = orders.get(order_id)
        if order:
            return {"status": "success", "order": order}
        else:
            return {"status": "not_found", "message": f"Order {order_id} not found"}
```

### 4. Dataset Creation

```python
def create_tau_bench_dataset(domain: str = "retail") -> Dataset:
    """Create dataset from tau-bench tasks."""
    
    # Load original tau-bench tasks
    from tau_bench.envs.retail.tasks import tasks
    
    # Convert to verifiers format
    dataset_rows = []
    for task in tasks:
        row = {
            "prompt": [{"role": "system", "content": get_system_prompt(domain)}],
            "question": task["instruction"],
            "info": {
                "user_id": task["user_id"],
                "expected_actions": task["actions"],
                "annotator": task["annotator"]
            },
            "answer": "Task completed successfully",  # Placeholder
            "task": f"tau_bench_{domain}"
        }
        dataset_rows.append(row)
    
    return Dataset.from_list(dataset_rows)
```

### 5. Rubric with Policy Compliance

```python
def create_tau_bench_rubric(policies: Dict[str, str]) -> vf.Rubric:
    """Create rubric that checks policy compliance and task completion."""
    
    def check_task_completion(completion, info, state, **kwargs) -> float:
        """Check if expected actions were taken."""
        expected_actions = info.get("expected_actions", [])
        actual_actions = state.get("tool_calls", [])
        
        # Compare actions
        if len(actual_actions) < len(expected_actions):
            return 0.0
            
        # Check each expected action
        score = 0.0
        for expected in expected_actions:
            found = False
            for actual in actual_actions:
                if (actual["name"] == expected["name"] and 
                    actual["arguments"] == expected["arguments"]):
                    found = True
                    break
            if found:
                score += 1.0 / len(expected_actions)
        
        return score
    
    def check_policy_compliance(completion, info, state, **kwargs) -> float:
        """Check if agent followed policies."""
        violations = state.get("policy_violations", [])
        if violations:
            return 0.0
        return 1.0
    
    rubric = vf.Rubric(
        funcs=[check_task_completion, check_policy_compliance],
        weights=[0.7, 0.3]  # 70% task completion, 30% policy compliance
    )
    
    return rubric
```

### 6. Complete Environment Loader

```python
def load_environment(
    domain: str = "retail",
    num_train_examples: int = 100,
    num_eval_examples: int = 15,
    user_model: str = "gpt-4.1-mini",
    **kwargs
) -> vf.Environment:
    """Load τ-bench environment."""
    
    # Create dataset
    full_dataset = create_tau_bench_dataset(domain)
    
    # Split into train/eval
    train_dataset = full_dataset.select(range(num_train_examples))
    eval_dataset = full_dataset.select(range(num_train_examples, num_train_examples + num_eval_examples))
    
    # Create tools
    tools = [
        FindUserByNameZipTool(),
        GetOrderDetailsTool(),
        GetProductDetailsTool(),
        ExchangeDeliveredOrderItemsTool(),
        # Add more tools as needed
    ]
    
    # Load policies
    policies = load_domain_policies(domain)
    
    # Create rubric
    rubric = create_tau_bench_rubric(policies)
    
    # Create environment
    env = TauBenchRetailEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        tools=tools,
        user_model=user_model,
        policies=policies,
        **kwargs
    )
    
    return env
```

## Migration Steps

1. **Set up directory structure**:
   ```
   environments/tau_bench_retail/
   ├── __init__.py
   ├── tau_bench_retail.py
   ├── tools.py
   ├── data/
   │   ├── policies.json
   │   ├── products.json
   │   └── users.json
   └── pyproject.toml
   ```

2. **Port data files** from original tau-bench

3. **Implement tools** one by one, testing each

4. **Create user simulator** that follows scenarios

5. **Add evaluation metrics** (pass^k reliability)

6. **Test with baseline models** (GPT-4, Claude)

## Key Differences to Handle

1. **User Simulation**: τ-bench uses LLM-based user simulation. In verifiers, we'll handle this in `env_response`

2. **Database State**: τ-bench tracks DB changes. We'll use the `state` dict to track this

3. **Policy Documents**: Store as JSON/text files and load during environment initialization

4. **Reliability Metric**: Implement pass^k by running multiple episodes with same task

## Testing Plan

1. Start with single task verification
2. Test tool execution accuracy
3. Validate user simulation responses
4. Check policy compliance detection
5. Run full benchmark comparison with original results

## Next Actions

1. Create the directory structure
2. Port the retail domain data
3. Implement core tools (5-10 most common)
4. Test with a simple scenario
5. Gradually add complexity