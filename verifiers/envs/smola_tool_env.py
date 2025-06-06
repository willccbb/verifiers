import inspect
import json
from typing import List, Dict, Any, Callable, Optional, Type

from datasets import Dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers.smola_parser import SmolaParser
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics.smola_tool_rubric import SmolaToolRubric

class SmolaToolEnv(MultiTurnEnv):
    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 tools: List[Any] = [],
                 system_prompt: str = DEFAULT_TOOL_PROMPT_TEMPLATE,
                 few_shot: List[Dict[str, str]] = [],
                 mask_env_response: bool = True,
                 max_steps: int = 10, **kwargs):
        # Format the system prompt with tool descriptions
        tool_descriptions = self._format_tool_descriptions(tools)
        formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            sampling_args=sampling_args,
            **kwargs
        )
        self.dataset_name = dataset
        self.max_steps = max_steps
        self.tools = {tool.name: tool for tool in tools}
        self.rubric = SmolaToolRubric(tools=tools)
        self.llm_parser = SmolaParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = SmolaParser(fields=["result"])

    def _format_tool_descriptions(self, tools: List[Any]) -> str:
        """Formats tool schemas into a user-friendly description string."""
        descriptions = []
        for tool in tools:
            desc = [f"{tool.name}: {tool.description}"]
            
            desc.append("\nArguments:")
            for arg_name, arg_info in tool.inputs.items():
                desc.append(f"  - {arg_name}: {arg_info['description']}")
            
            desc.append(f"\nReturns: {tool.output_type}")
            
            descriptions.append("\n".join(desc))
        
        return "\n\n".join(descriptions)

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        step_count = 0
        
        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
        return step_count
    
    def is_completed(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs: Any) -> bool:
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count > self.max_steps:
                return True
            
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def call_tool(self, tool_json: str, **kwargs: Any) -> str:
        """Call a SmolaAgents Tool object based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return "Error: Tool command must be a JSON object, e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
            tool_name = command.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name', e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'. " + "Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
            tool = self.tools[tool_name]
            tool_args = command.get("args", {})
            if isinstance(tool_args, str):
                return f"Error: Arguments for {tool_name} must be a JSON object matching the tool's input schema, not a string." 
            
            # Call the tool object with arguments
            result = tool(**tool_args)
            return str(result)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format. Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
        except Exception as e:
            return f"Error: {str(e)}. " + "Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"

    def env_response(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs: Any) -> Dict[str, str]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    return {"role": "user", "content": self.env_parser.format(result=result)}, {}
                else:
                    return {"role": "user", "content": "Error: Tool execution returned empty output."}, {}
        except Exception:
            pass
        return {"role": "user", "content": "Error: Tool command not found or invalid XML format. Please ensure correct formatting."}, {}