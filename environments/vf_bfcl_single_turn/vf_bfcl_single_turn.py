import json
from typing import List, Dict, Any, Optional
import random
import re

from datasets import load_dataset, Dataset

import verifiers as vf


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    """Format tools for the system prompt."""
    return json.dumps(tools, indent=2)


def load_environment(
    num_train_examples: int = 5000, 
    num_eval_examples: int = 500,
    **kwargs
) -> vf.SingleTurnEnv:
    """
    Loads the BFCL v3 single turn function calling environment with reward shaping.
    """
    # Load the BFCL dataset
    # Try multiple potential sources for BFCL data
    dataset_loaded = False
    dataset = None
    
    try:
        # Try loading from gorilla-llm repository
        dataset = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="test")
        dataset_loaded = True
    except:
        pass
    
    if not dataset_loaded:
        try:
            # Try loading from llamastack repository
            dataset = load_dataset("llamastack/bfcl_v3", split="train")
            dataset_loaded = True
        except:
            pass
    
    if not dataset_loaded:
        # If no dataset available, create synthetic data similar to tool_test
        dataset = create_synthetic_function_calling_dataset(num_train_examples + num_eval_examples)
    
    # Process examples for training
    def process_example(example):
        # Extract the query and tools from the example
        if isinstance(example, dict):
            query = example.get("question", example.get("query", example.get("user_query", "")))
            tools = example.get("functions", example.get("tools", []))
            expected_output = example.get("expected_output", example.get("answers", example.get("function_calls", [])))
        else:
            # Handle synthetic data
            query = example["query"]
            tools = example["tools"]
            expected_output = example["expected_calls"]
        
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query) if query is not None else ""
        
        # Ensure tools is a list
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except:
                tools = []
        elif not isinstance(tools, list):
            tools = [] if tools is None else [tools]
        
        # Ensure each tool is a dict
        cleaned_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                cleaned_tools.append(tool)
            elif isinstance(tool, str):
                try:
                    parsed_tool = json.loads(tool)
                    if isinstance(parsed_tool, dict):
                        cleaned_tools.append(parsed_tool)
                except:
                    continue
        tools = cleaned_tools
        
        # Format the prompt with a clear example
        system_prompt = f"""You are a helpful assistant with access to the following functions:

{format_tools_for_prompt(tools)}

To use these functions, you must respond with a JSON array of function calls inside <function_calls> tags.
Each function call should have "name" and "arguments" fields.

Example format:
<function_calls>[{{"name": "function_name", "arguments": {{"param1": "value1"}}}}]</function_calls>

Always use this exact format with the <function_calls> tags."""
        
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Parse expected calls
        if isinstance(expected_output, str):
            try:
                expected_calls = json.loads(expected_output)
            except:
                expected_calls = []
        else:
            expected_calls = expected_output if expected_output else []
        
        # Ensure expected_calls is a list
        if isinstance(expected_calls, dict):
            expected_calls = [expected_calls]
        elif not isinstance(expected_calls, list):
            expected_calls = [] if expected_calls is None else [expected_calls]
        
        # Ensure each expected call is a dict
        cleaned_calls = []
        for call in expected_calls:
            if isinstance(call, dict):
                cleaned_calls.append(call)
            elif isinstance(call, str):
                try:
                    parsed_call = json.loads(call)
                    if isinstance(parsed_call, dict):
                        cleaned_calls.append(parsed_call)
                except:
                    continue
        expected_calls = cleaned_calls
        
        return {
            "prompt": prompt,
            "info": {
                "expected_calls": expected_calls,
                "tools": tools
            }
        }
    
    # Process dataset
    if isinstance(dataset, Dataset):
        processed_data = [process_example(ex) for ex in dataset]
    else:
        processed_data = [process_example(ex) for ex in dataset]
    
    # Shuffle data
    random.shuffle(processed_data)
    
    # Split into train and eval
    total_examples = len(processed_data)
    train_size = min(num_train_examples, int(0.9 * total_examples))
    eval_size = min(num_eval_examples, total_examples - train_size)
    
    train_data = processed_data[:train_size]
    eval_data = processed_data[train_size:train_size + eval_size]
    
    # Create datasets with error handling
    try:
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
    except Exception as e:
        print(f"Error creating datasets from processed data: {e}")
        print("Falling back to synthetic data...")
        # Create synthetic data as fallback
        synthetic_data = create_synthetic_function_calling_dataset(num_train_examples + num_eval_examples)
        processed_synthetic = [process_example(ex) for ex in synthetic_data]
        
        # Split synthetic data
        train_size = min(num_train_examples, int(0.9 * len(processed_synthetic)))
        train_data = processed_synthetic[:train_size]
        eval_data = processed_synthetic[train_size:train_size + num_eval_examples]
        
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
    
    # Create parser for function calls
    parser = vf.XMLParser(
        fields=["function_calls"],
        answer_field="function_calls"
    )
    
    # Define reward function with reward shaping
    def function_call_reward(completion, info):
        """Reward function with shaping to help model learn the format."""
        try:
            # Get the completion content
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[-1].get('content', '')
            else:
                return 0.0
            
            # Debug logging
            if not hasattr(function_call_reward, 'call_count'):
                function_call_reward.call_count = 0
            function_call_reward.call_count += 1
            
            if function_call_reward.call_count <= 5:
                print(f"\n[Reward {function_call_reward.call_count}] Content: {content[:100]}...")
            
            # Reward shaping - give partial credit for progress
            reward = 0.0
            
            # Check for XML tags (highest reward component)
            has_opening_tag = '<function_calls>' in content
            has_closing_tag = '</function_calls>' in content
            has_both_tags = has_opening_tag and has_closing_tag
            
            if has_both_tags:
                reward += 0.3  # Major reward for using correct format
            elif has_opening_tag or has_closing_tag:
                reward += 0.1  # Partial reward for attempting tags
            
            # Check for JSON structure
            has_json_array = '[' in content and ']' in content
            has_json_object = '{' in content and '}' in content
            has_name_field = '"name"' in content
            has_arguments_field = '"arguments"' in content
            
            if has_json_array:
                reward += 0.1
            if has_json_object:
                reward += 0.1
            if has_name_field:
                reward += 0.1
            if has_arguments_field:
                reward += 0.1
            
            # Try to parse with the XML parser
            parsed_calls = parser.parse_answer(completion)
            
            if parsed_calls:
                reward += 0.3  # Successfully parsed!
                
                try:
                    # Parse the JSON
                    called_functions = json.loads(parsed_calls)
                    if isinstance(called_functions, dict):
                        called_functions = [called_functions]
                    
                    expected_calls = info.get("expected_calls", [])
                    
                    # Normalize and compare
                    def normalize_call(call):
                        return {
                            "name": call.get("name", ""),
                            "arguments": call.get("arguments", {})
                        }
                    
                    if expected_calls:
                        normalized_called = [normalize_call(c) for c in called_functions]
                        normalized_expected = [normalize_call(c) for c in expected_calls]
                        
                        # Check for correct function names
                        expected_names = {c["name"] for c in normalized_expected}
                        called_names = {c["name"] for c in normalized_called}
                        
                        # Reward for calling correct functions
                        correct_names = expected_names.intersection(called_names)
                        if correct_names:
                            reward += 0.2 * (len(correct_names) / len(expected_names))
                        
                        # Full reward for exact match
                        matches = 0
                        for expected in normalized_expected:
                            for called in normalized_called:
                                if (expected["name"] == called["name"] and 
                                    expected["arguments"] == called["arguments"]):
                                    matches += 1
                                    break
                        
                        if matches == len(normalized_expected) and len(normalized_called) == len(normalized_expected):
                            reward = 1.0  # Perfect match!
                        else:
                            # Add match bonus
                            reward += 0.2 * (matches / len(normalized_expected))
                    else:
                        # No expected calls - reward for not making calls
                        if not called_functions:
                            reward = 1.0
                        
                except:
                    pass  # JSON parsing failed, but we still have partial rewards
            
            # Log reward breakdown for debugging
            if function_call_reward.call_count <= 5:
                print(f"[Reward {function_call_reward.call_count}] Total: {reward:.3f}")
                print(f"  Tags: {has_both_tags}, JSON: {has_json_array or has_json_object}")
                print(f"  Parsed: {parsed_calls is not None}")
            
            return min(reward, 1.0)  # Cap at 1.0
            
        except Exception as e:
            print(f"Error in reward function: {e}")
            return 0.0
    
    # Create rubric with proper reward function
    rubric = vf.Rubric(
        funcs=[function_call_reward],
        weights=[1.0]
    )
    
    # Create environment
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
    )
    
    return vf_env


def create_synthetic_function_calling_dataset(num_examples: int) -> List[Dict[str, Any]]:
    """Create synthetic function calling examples if real dataset is not available."""
    
    # Define some example functions
    example_functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "num_results": {"type": "integer", "description": "Number of results to return", "default": 5}
                },
                "required": ["query"]
            }
        },
        {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to evaluate"}
                },
                "required": ["expression"]
            }
        },
        {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Email recipient"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    ]
    
    # Example queries and their expected function calls
    query_templates = [
        {
            "query": "What's the weather like in {city}?",
            "tools": [example_functions[0]],
            "expected_calls": [{"name": "get_weather", "arguments": {"location": "{city}"}}]
        },
        {
            "query": "Search for information about {topic}",
            "tools": [example_functions[1]],
            "expected_calls": [{"name": "search_web", "arguments": {"query": "{topic}"}}]
        },
        {
            "query": "Calculate {expression}",
            "tools": [example_functions[2]],
            "expected_calls": [{"name": "calculate", "arguments": {"expression": "{expression}"}}]
        },
        {
            "query": "Send an email to {email} about {subject}",
            "tools": [example_functions[3]],
            "expected_calls": [{"name": "send_email", "arguments": {"to": "{email}", "subject": "{subject}", "body": "This is an automated message about {subject}."}}]
        },
        {
            "query": "What's the weather in {city} and search for hotels there",
            "tools": [example_functions[0], example_functions[1]],
            "expected_calls": [
                {"name": "get_weather", "arguments": {"location": "{city}"}},
                {"name": "search_web", "arguments": {"query": "hotels in {city}"}}
            ]
        }
    ]
    
    # Generate examples
    cities = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ"]
    topics = ["machine learning", "climate change", "space exploration", "quantum computing", "renewable energy"]
    expressions = ["2 + 2", "10 * 5", "100 / 4", "7^2", "sqrt(16)"]
    emails = ["john@example.com", "jane@example.com", "admin@company.com", "support@service.com"]
    subjects = ["Meeting Update", "Project Status", "Weekly Report", "Important Notice"]
    
    dataset = []
    for i in range(num_examples):
        template = random.choice(query_templates)
        
        # Fill in the template
        query = template["query"]
        expected_calls = template["expected_calls"]
        
        # Replace placeholders
        city = random.choice(cities)
        topic = random.choice(topics)
        expression = random.choice(expressions)
        email = random.choice(emails)
        subject = random.choice(subjects)
        
        query = query.replace("{city}", city)
        query = query.replace("{topic}", topic)
        query = query.replace("{expression}", expression)
        query = query.replace("{email}", email)
        query = query.replace("{subject}", subject)
        
        # Update expected calls
        updated_calls = []
        for call in expected_calls:
            updated_call = {
                "name": call["name"],
                "arguments": {}
            }
            for key, value in call["arguments"].items():
                updated_value = value
                updated_value = updated_value.replace("{city}", city)
                updated_value = updated_value.replace("{topic}", topic)
                updated_value = updated_value.replace("{expression}", expression)
                updated_value = updated_value.replace("{email}", email)
                updated_value = updated_value.replace("{subject}", subject)
                updated_call["arguments"][key] = updated_value
            updated_calls.append(updated_call)
        
        dataset.append({
            "query": query,
            "tools": template["tools"],
            "expected_calls": updated_calls
        })
    
    return dataset