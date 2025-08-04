import random

from datasets import Dataset

import verifiers as vf


# dummy tools for sanity checking parallel tool calls
def tool_A(x: int) -> int:
    """
    Tool for adding 1 to an integer.

    Args:
        x: The integer to add 1 to.

    Returns:
        The integer plus 1.
    """
    return x + 1


def tool_B(x: str) -> str:
    """
    Tool for concatenating a string with "2".

    Args:
        x: The string to concatenate with "2".

    Returns:
        The string concatenated with "2".
    """
    return x + "2"


def tool_C(x: float) -> float:
    """
    Tool for adding 3.0 to a float.

    Args:
        x: The float to add 3.0 to.

    Returns:
        The float plus 3.0.
    """
    return x + 3.0


def tool_D(x: bool) -> bool:
    """
    Tool for negating a boolean.

    Args:
        x: The boolean to negate.

    Returns:
        The negated boolean.
    """
    return not x


tool_list = [tool_A, tool_B, tool_C, tool_D]
tool_name_list = [tool.__name__ for tool in tool_list]


def tool_call_reward_func(completion, info):
    """Improved reward function that gives partial credit and detailed feedback."""
    try:
        # Extract tool calls from the completion
        tool_calls = completion[-1].get("tool_calls", [])
        called_tool_names = sorted([call.function.name for call in tool_calls])
        expected_tool_names = sorted(info["tool_names"])
        
        # Perfect match gets full reward
        if called_tool_names == expected_tool_names:
            return 1.0
        
        # No tool calls when expected gets zero
        if not called_tool_names and expected_tool_names:
            return 0.0
            
        # Unexpected tool calls when none expected gets zero  
        if called_tool_names and not expected_tool_names:
            return 0.0
        
        # Partial credit for overlapping tools
        if called_tool_names and expected_tool_names:
            # Jaccard similarity for partial credit
            called_set = set(called_tool_names)
            expected_set = set(expected_tool_names)
            intersection = called_set.intersection(expected_set)
            union = called_set.union(expected_set)
            return len(intersection) / len(union) if union else 0.0
        
        return 0.0
    except Exception as e:
        return 0.0


def tool_call_precision_func(completion, info):
    """Reward function measuring precision: correct calls / total calls."""
    try:
        tool_calls = completion[-1].get("tool_calls", [])
        called_tool_names = [call.function.name for call in tool_calls]
        expected_tool_names = set(info["tool_names"])
        
        if not called_tool_names:
            return 1.0 if not expected_tool_names else 0.0
            
        correct_calls = sum(1 for name in called_tool_names if name in expected_tool_names)
        return correct_calls / len(called_tool_names)
    except Exception:
        return 0.0


def tool_call_recall_func(completion, info):
    """Reward function measuring recall: correct calls / expected calls."""
    try:
        tool_calls = completion[-1].get("tool_calls", [])
        called_tool_names = set([call.function.name for call in tool_calls])
        expected_tool_names = set(info["tool_names"])
        
        if not expected_tool_names:
            return 1.0 if not called_tool_names else 0.0
            
        correct_calls = len(called_tool_names.intersection(expected_tool_names))
        return correct_calls / len(expected_tool_names)
    except Exception:
        return 0.0


def argument_quality_func(completion, info):
    """Reward function for checking if tool arguments are reasonable."""
    try:
        tool_calls = completion[-1].get("tool_calls", [])
        if not tool_calls:
            return 0.0
            
        quality_score = 0.0
        total_calls = 0
        
        for call in tool_calls:
            total_calls += 1
            try:
                # Check if arguments are provided and non-empty
                args = call.function.arguments
                if args and isinstance(args, dict) and len(args) > 0:
                    # Check if argument types seem reasonable
                    has_valid_args = False
                    for key, value in args.items():
                        if value is not None and str(value).strip():
                            has_valid_args = True
                            break
                    if has_valid_args:
                        quality_score += 1.0
            except Exception:
                continue
                
        return quality_score / total_calls if total_calls > 0 else 0.0
    except Exception:
        return 0.0


def load_environment(
    num_train_examples: int = 1000, num_eval_examples: int = 100, 
    difficulty_curriculum: bool = True
) -> vf.ToolEnv:
    """
    Loads an improved custom environment with curriculum learning and better prompts.
    """

    train_rows = []
    eval_rows = []
    
    # Generate more diverse and challenging prompts
    prompt_templates = [
        "Call the following tools with arguments of your choice: {tool_names}",
        "Please invoke these tools: {tool_names}",
        "Use these functions: {tool_names}",
        "Execute the following tools: {tool_names}",
        "I need you to call: {tool_names}",
    ]
    
    for i in range(num_train_examples + num_eval_examples):
        # Curriculum learning: start with fewer tools, gradually increase complexity
        if difficulty_curriculum and i < num_train_examples:
            # Early examples: 1-2 tools, later examples: up to all tools
            progress = i / num_train_examples
            max_tools = 1 + int(progress * (len(tool_name_list) - 1))
            min_tools = 1 if progress < 0.3 else 1 + int((progress - 0.3) * 2)
            min_tools = min(min_tools, max_tools)
        else:
            # Evaluation: full complexity
            min_tools = 1
            max_tools = len(tool_name_list)
            
        num_tools = random.randint(min_tools, max_tools)
        tool_names = random.sample(tool_name_list, num_tools)
        
        # Vary prompt templates for robustness
        template = random.choice(prompt_templates)
        
        prompt = [
            {
                "role": "user",
                "content": template.format(tool_names=tool_names),
            }
        ]
        info = {"tool_names": tool_names}
        
        if i < num_train_examples:
            train_rows.append({"prompt": prompt, "info": info})
        else:
            eval_rows.append({"prompt": prompt, "info": info})

    dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)
    parser = vf.Parser()
    
    # Enhanced rubric with multiple reward components
    rubric = vf.ToolRubric(tools=tool_list)
    
    # Primary reward function with higher weight
    rubric.add_reward_func(tool_call_reward_func, weight=1.0)
    
    # Additional reward functions for better signal
    rubric.add_reward_func(tool_call_precision_func, weight=0.3) 
    rubric.add_reward_func(tool_call_recall_func, weight=0.3)
    rubric.add_reward_func(argument_quality_func, weight=0.2)
    
    vf_env = vf.ToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
        tools=tool_list,
        max_turns=1,
    )
    return vf_env
