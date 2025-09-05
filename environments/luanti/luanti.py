"""
Luanti (Minetest) Code Generation Environment for Verifiers

Evaluates LLM capability to generate, fix, and modify Luanti/Minetest mod code
across scaffold, repair, refactor, and documentation tasks.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import verifiers as vf


# Sample dataset for testing (embedded)
SAMPLE_TASKS = [
    {
        "instruction": "Create a Luanti node that emits light level 10 and drops itself when dug.",
        "any_of": ["light_source", "minetest.register_node"],
        "task_type": "scaffold"
    },
    {
        "instruction": "Fix this broken node code: minetest.register_node('test', {description = 'Test'})",
        "any_of": ["tiles", "groups"],
        "task_type": "repair"
    },
    {
        "instruction": "Modify this node to change light level from 8 to 14: minetest.register_node('lamp', {light_source = 8})",
        "any_of": ["light_source = 14"],
        "task_type": "refactor"
    }
]


def extract_lua_code(response: str) -> str:
    """Extract Lua code from LLM response"""
    if "### Response:" in response:
        response = response.split("### Response:")[-1]
    
    # Look for code blocks or minetest patterns
    patterns = [
        r'```(?:lua)?\n(.*?)```',
        r'minetest\.register_\w+\([^}]+}\)',
        r'function\s+\w+.*?end'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
    
    return response.strip()


def check_luanti_syntax(code: str) -> float:
    """Check basic Luanti/Lua syntax correctness"""
    score = 0.0
    
    # Has minetest API call
    if re.search(r'minetest\.(register_\w+|\w+)', code):
        score += 0.4
    
    # Balanced braces
    if _check_balanced_braces(code):
        score += 0.3
    
    # No obvious syntax errors
    errors = [r'\{\s*,', r',\s*}', r'=\s*,', r',,']
    if not any(re.search(err, code) for err in errors):
        score += 0.3
        
    return score


def check_luanti_api_usage(code: str, task: dict) -> float:
    """Check correct Minetest API usage"""
    score = 0.0
    
    # Correct registration call
    if re.search(r'minetest\.register_(node|tool|craftitem|entity|craft)\s*\(', code):
        score += 0.4
    
    # Has required properties
    required_props = ["description", "tiles", "groups", "light_source", "drop"]
    found = sum(1 for prop in required_props if prop in code)
    if found >= 2:
        score += 0.4
    
    # Proper table structure
    if re.search(r'\{[^{}]*description\s*=.*?\}', code, re.DOTALL):
        score += 0.2
        
    return score


def check_task_completion(code: str, task: dict) -> float:
    """Check if task requirements were met"""
    score = 0.0
    
    # Check expected patterns
    expected = task.get("any_of", [])
    if expected:
        matches = sum(1 for pattern in expected if re.search(pattern, code, re.IGNORECASE))
        if matches > 0:
            score += 0.8
    else:
        score += 0.4  # Default partial credit
    
    # Check forbidden patterns
    forbidden = task.get("must_not_contain", [])
    violations = sum(1 for pattern in forbidden if re.search(pattern, code, re.IGNORECASE))
    if violations == 0:
        score += 0.2
        
    return min(1.0, score)


def _check_balanced_braces(code: str) -> bool:
    """Check if braces/parentheses are balanced"""
    stack = []
    pairs = {'(': ')', '{': '}', '[': ']'}
    
    for char in code:
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                return False
    
    return len(stack) == 0


def luanti_reward_func(parser, completion: str, task_data: dict, **kwargs) -> float:
    """
    Reward function for Luanti code generation
    
    Scoring breakdown:
    - Syntax (30%): Valid Lua, balanced braces, no syntax errors
    - API Usage (40%): Correct minetest calls, proper properties
    - Task Completion (30%): Addresses prompt requirements
    """
    code = extract_lua_code(completion)
    
    # Component scores
    syntax_score = check_luanti_syntax(code) * 0.3
    api_score = check_luanti_api_usage(code, task_data) * 0.4
    task_score = check_task_completion(code, task_data) * 0.3
    
    total_score = syntax_score + api_score + task_score
    return min(1.0, max(0.0, total_score))


def load_environment(
    dataset_path: Optional[str] = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    use_sample_data: bool = False
):
    """
    Load Luanti code generation environment
    
    Args:
        dataset_path: Path to JSONL dataset (if None, uses sample data)
        num_train_examples: Limit training examples (-1 for all)
        num_eval_examples: Limit eval examples (-1 for all)  
        use_sample_data: Use embedded sample tasks for testing
    """
    
    # Load dataset
    if use_sample_data or dataset_path is None:
        # Use embedded sample data for testing
        dataset_items = SAMPLE_TASKS
    else:
        # Load from JSONL file
        dataset_items = []
        try:
            with open(dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        dataset_items.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: Dataset {dataset_path} not found, using sample data")
            dataset_items = SAMPLE_TASKS
    
    # Apply limits
    if num_eval_examples > 0:
        dataset_items = dataset_items[:num_eval_examples]
    
    # Convert to verifiers format
    dataset = []
    for item in dataset_items:
        # Create verifiers-compatible dataset item
        prompt = item.get("instruction", item.get("prompt", ""))
        dataset.append({
            "prompt": prompt,
            "task_data": item,  # Pass full item for reward function
            "answer": item.get("output", "")  # Expected output if available
        })
    
    # Use simple parser for code extraction
    parser = vf.Parser(extract_fn=extract_lua_code)
    
    # Create rubric with our custom reward function
    rubric = vf.Rubric(
        parser=parser,
        funcs=[luanti_reward_func],
        weights=[1.0],
    )
    
    # System prompt optimized for Luanti code generation
    system_prompt = """You are an expert Luanti (Minetest) mod developer. Generate clean, functional Lua code using the Minetest API. Focus on:

- Correct minetest.register_* function usage
- Proper table structure with required properties
- Valid light_source values (0-14)
- Appropriate groups, tiles, and descriptions
- Clean, readable code formatting

Respond with just the Lua code, no explanations unless requested."""

    # Create the environment
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=dataset,  # Same dataset for now
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    
    return vf_env