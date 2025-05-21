from abc import abstractmethod
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random
import time
from typing import List, Dict, Sequence, Any, Tuple

from datasets import Dataset
from pydantic import BaseModel
from openai import OpenAI

from ..imports import SamplingParams  # type: ignore
from verifiers.parsers import Parser
from verifiers.rubrics import Rubric
from verifiers.envs.environment import Environment
from verifiers.utils import format_dataset

class ChatOutput(BaseModel):
    token_ids: List[int]
    text: str

class ChatResponseItem(BaseModel):
    prompt_token_ids: List[int]
    outputs: List[ChatOutput]

class ChatResponse(BaseModel):
    responses: List[ChatResponseItem]

def dict_to_chat_response(data: Dict[str, Any]) -> ChatResponse:
    """
    Recursively convert a dictionary to a ChatResponse object
    """
    # First, convert all outputs to ChatOutput objects
    if "responses" in data:
        for i, response_item in enumerate(data["responses"]):
            if "outputs" in response_item:
                data["responses"][i]["outputs"] = [
                    ChatOutput(**output) for output in response_item["outputs"]
                ]
        
        # Then convert all response items to ChatResponseItem objects
        data["responses"] = [ChatResponseItem(**item) for item in data["responses"]]
    
    # Finally, convert the entire dict to a ChatResponse object
    return ChatResponse(**data)

class MultiTurnEnv(Environment):
    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 system_prompt: str | None = None,
                 few_shot: List[Dict[str, str]] | None = None,
                 parser: Parser = Parser(),
                 rubric: Rubric = Rubric(),
                 sampling_args: Dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 10,
                 max_steps: int = 10,
                 sleep_time: float = 1.0,
                 **kwargs):
        
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=parser,
            rubric=rubric,
            **kwargs
        )
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        if dataset is not None:
            self.dataset = format_dataset(
                dataset=dataset,
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
        else:
            self.dataset = None
        if eval_dataset is not None:
            self.eval_dataset = format_dataset(
                dataset=eval_dataset,
                system_prompt=self.system_prompt,
                few_shot=few_shot
            )
        else:   
            self.eval_dataset = None
        self.sampling_args = {
            "extra_body": {
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
            },
        }
        self.sampling_args.update(sampling_args)
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
        self.sleep_time = sleep_time
        self.max_steps = max_steps
        # TODO: remove these
        self.tokenizer = None
        self.eot_id = 151643
        self.message_end_id = 151645

    def get_dataset(self, n: int = -1, seed: int = 0, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.dataset is not None:
            return self.dataset.shuffle(seed=seed).select(range(n)) # type: ignore
        return self.dataset

    def get_eval_dataset(self, n: int = -1, seed: int = 0, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.eval_dataset is not None:
            return self.eval_dataset.shuffle(seed=seed).select(range(n)) # type: ignore
        return self.eval_dataset

    @abstractmethod
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        pass

    """
    Reward logic:

    Rubric has get_reward_funcs and get_reward_weights

    """

    def step(self,
             states: List[Dict[str, Any]],
             client: OpenAI,
             sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]

        llm_responses = client.chat.completions.create(
            messages=messages_to_step,
            n=1,
            repetition_penalty=sampling_params.repetition_penalty,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,        
            top_k=sampling_params.top_k,
            min_p=sampling_params.min_p,
            max_tokens=sampling_params.max_tokens, # type: ignore
            stop=sampling_params.stop, # type: ignore
            include_stop_str_in_output=sampling_params.include_stop_str_in_output,
            skip_special_tokens=sampling_params.skip_special_tokens,
            spaces_between_special_tokens=sampling_params.spaces_between_special_tokens
        ) # type: ignore
        llm_responses = dict_to_chat_response(llm_responses).responses

        def update_state(j, llm_response):
            # sleep for 0-1 seconds to avoid rate limiting
            time.sleep(self.sleep_time * random.random())

            state = deepcopy(states[j])
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = llm_response.prompt_token_ids
            state["messages"].append({"role": "assistant", "content": llm_response.outputs[0].text})
        
            # get token lengths of env response and new completion
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len  = len(list(llm_response.prompt_token_ids)) - total_prev_len # type: ignore
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # update completion masks
            state["completion_mask"].extend([self.env_mask] * env_response_len)
            state["completion_mask"].extend([1] * new_completion_len)

            # update completion ids
            state["completion_ids"] = list(llm_response.prompt_token_ids) # type: ignore
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][len(state["prompt_ids"]):]

            if state["completion_ids"][-1] != 198 and state["completion_ids"][-2] != self.message_end_id:
                state["completion_ids"].append(self.message_end_id)
                state["completion_ids"].append(198)
                state["completion_mask"].append(1)
                state["completion_mask"].append(1)

            if len(state["completion_ids"]) > len(state["completion_mask"]): # type: ignore
                state["completion_mask"].extend([1] * (len(state["completion_ids"]) - len(state["completion_mask"]))) # type: ignore
            if len(state["completion_mask"]) > len(state["completion_ids"]): # type: ignore
                state["completion_mask"] = state["completion_mask"][:len(state["completion_ids"])] # type: ignore
            
            if self.is_completed(state["messages"]) or len(state["completion_ids"]) > sampling_params.max_tokens - 1: # type: ignore
                state["completed"] = True

                # ids and mask should be the same length
                state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                state["completion_mask"] = state["completion_mask"][:len(state["completion_ids"])]

                # calculate rewards

            else:
                state["messages"].append(self.env_response(state["messages"]))

            # enforce that the completion mask and completion ids are the same length
            # weird bug that happens rarely and only for certain models; something tokenizer related :(
            if not len(state["completion_mask"]) == len(state["completion_ids"]):
                print(state["messages"])
                print(state["completion_mask"])
                print(state["completion_ids"])
                min_len = min(len(state["completion_mask"]), len(state["completion_ids"]))
                state["completion_mask"] = state["completion_mask"][:min_len]
                state["completion_ids"] = state["completion_ids"][:min_len]

            return j, state

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda args: update_state(*args),
                [(j, llm_responses[i]) for i, j in enumerate(live_indices)]
            ))

        for j, state in results:
            states[j] = state

        return states

    def prepare_initial_state(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the initial state for rollout.
        """
        messages = []
        if isinstance(input["prompt"], str):
            messages = [{"role": "user", "content": input["prompt"]}]
        else:
            messages = input["prompt"]

        state = {
            "messages": messages,
            "prompt_messages": deepcopy(messages),
            "completion_messages": [],
            "num_prompt_messages": len(messages),
            "num_completion_messages": 0,
            "prompt_ids": [],
            "completed": False,
            "completion_ids": [],
            "completion_mask": [],
            "completion_rewards": []
        }
        # extend state with additional kwargs, error if duplicate keys
        state_keys = list(state.keys())
        for k, v in input.items():
            if k in state:
                raise ValueError(f"Warning: duplicate key {k} in input. Prohibited keys: {state_keys}")
            else:
                state[k] = v
        return state

    def generate(self,
                 #inputs: Dict[str, Any],
                 prompts: List[List[Dict[str, Any]]] | List[str],
                 client: OpenAI,  
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] |  List[List[Dict[str, Any]]]]:
        """
        Generate rollouts for a set of inputs.
        Input:
        - inputs dict with keys:
            - prompt (already duplicated for GRPO group)
            - answer
            - task (optional)
            - **kwargs: additional kwargs

        - vLLM client object
        - sampling_params object

        output:
        """
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states = [self.prepare_initial_state(input) for input in inputs]

        # main loop
        while not all_completed:
            states = self.step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)

        completion_messages = [s["messages"][s["prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        completion_rewards = [s["completion_rewards"] for s in states]
        
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "rewards": completion_rewards,
            "mask": completion_mask
        }
        return output

    def step_api(self, 
             client: Any,
             model: str,
             messages: List[Dict[str, str]],
             sampling_args: Dict[str, Any] = {},
             **kwargs: Any) -> Tuple[List[Dict[str, str]], bool]:
        """
        Execute a single step using OpenAI API, including environment response if needed.
        
        Args:
            client: OpenAI client instance
            messages: Conversation history
            model: Model name to use
            **kwargs: Additional arguments for the chat completion API
        
        Returns:
            Updated messages list with assistant response and possibly environment response
        """
        messages_copy = deepcopy(messages)
        
        try:            
            # Get assistant response
            response = client.chat.completions.create(
                model=model,
                messages=messages_copy,
                **sampling_args
            )
            # Add assistant response to messages
            assistant_msg = {
                "role": "assistant", 
                "content": response.choices[0].message.content
            }
            messages_copy.append(assistant_msg)
            
            # Check if we're done
            if self.is_completed(messages_copy):
                rollout_is_completed = True
            else:
                rollout_is_completed = False
                # If not done, get and add environment response
                env_msg = self.env_response(messages_copy)
                messages_copy.append(env_msg)
            return messages_copy, rollout_is_completed
            
        except Exception as e:
            # Handle errors by adding error message and returning
            error_msg = {"role": "assistant", "content": f"Error in API call: {str(e)}"}
            messages_copy.append(error_msg)
            return messages_copy, True
    
    def eval_api(self, 
                client: Any,
                model: str,
                max_concurrent: int = 32,
                num_samples: int = -1,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any):
        
        eval_sampling_args = deepcopy(self.sampling_args)
        eval_sampling_args.update(sampling_args)
        """
        Evaluate model using OpenAI API with async processing.
        
        Args:
            client: OpenAI client instance
            model: Model name as string
            max_concurrent: Maximum number of concurrent API calls
            timeout: Maximum seconds to wait for each example
            sampling_args: Arguments specific to sampling (separate from env sampling_args)
            **kwargs: Additional arguments for evaluation
        
        Returns:
            Tuple of (eval_dataset, rewards)
        """
        def run_evaluation():
            # Import libraries here to avoid requiring them for normal operation
            import asyncio
            from asyncio import Semaphore
            # Get the evaluation dataset
            if self.eval_dataset is None:
                self.logger.info("Eval dataset not provided, falling back to default dataset")
                self.eval_dataset = self.get_dataset(**kwargs)
                
            if self.eval_dataset is None:
                raise ValueError("Failed to load evaluation dataset")
            
            eval_dataset = self.eval_dataset
            if num_samples > 0:
                self.logger.info(f"Evaluating last {num_samples} examples")
                eval_dataset = eval_dataset.select(range(len(eval_dataset) - num_samples, len(eval_dataset)))

            async def process_example(example, semaphore) -> Dict[str, Any]:
                async with semaphore:
                    # Initialize conversation with system prompt and few-shot examples
                    prompt = example["prompt"]
                    messages = deepcopy(example["prompt"])
                    answer = example["answer"]
                    
                    # Save the length of initial messages to extract just the interaction part later
                    initial_length = len(messages)

                    # Run the conversation loop until completion or max steps
                    for _ in range(self.max_steps):  # Safety limit on conversation turns
                        try:
                            # step_api returns a tuple (messages, is_completed)
                            step_result = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.step_api(
                                    client=client,
                                    model=model,
                                    messages=messages,
                                    sampling_args=eval_sampling_args
                                )
                            )
                            
                            # Unpack the step_api result
                            messages, is_completed = step_result
                            
                            # If the rollout is completed, break the loop
                            if is_completed:
                                break
                            
                        except Exception as e:
                            print(f"Error processing example {example.get('id', 'unknown')}: {str(e)}")
                            break
                    
                    # Extract only the interaction part (not system/few-shot)
                    completion = messages[initial_length:]
                    
                    result = {
                        "prompt": prompt,
                        "completion": completion,
                        "answer": answer
                    }
                    if 'task' in example:
                        result['task'] = example['task']
                    else:
                        result['task'] = None
                    return result
            
            async def run_all_examples() -> List[Dict[str, Any]]:
                # Create semaphore for concurrency control
                from tqdm.asyncio import tqdm_asyncio

                semaphore = Semaphore(max_concurrent)
                
                # Process all examples concurrently
                tasks = [process_example(example, semaphore) for example in eval_dataset]
                results = await tqdm_asyncio.gather(
                    *tasks,
                    total=len(eval_dataset),
                    desc=f"Evaluating {len(eval_dataset)} examples"
                )
                
                return results
            
            # Run the async evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_all_examples())
            finally:
                loop.close()

            def flatten(results: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
                """
                Flatten a list of dictionaries into a single dictionary with lists of values.
                """
                return {k: [item[k] for item in results] for k in results[0]}

            results = flatten(results)
            results_rewards = self.rubric.score_rollout_group( 
                prompts=results['prompt'],
                completions=results['completion'],
                answers=results['answer'],
                tasks=results['task'],
                max_workers=self.max_workers
            )
            results_rewards = flatten(results_rewards)
            results.update(results_rewards)
            return results

        # Run the evaluation function
        return run_evaluation()
    
    def make_api_dataset(self,
                         client: Any,
                         model: str,
                         max_concurrent: int = 32,
                         num_samples: int = -1,
                         sampling_args: Dict[str, Any] = {"temperature": 0.6},
                         **kwargs: Any) -> Dataset:
        """
        Make a dataset from the evaluation results.
        """
        results = self.eval_api(
            client,
            model, 
            max_concurrent, 
            num_samples, 
            sampling_args
        )
        if results['task'][0] is not None:
            dataset = Dataset.from_dict({
                "prompt": results['prompt'],
                "completion": results['completion'],
                "answer": results['answer'],
                "reward": results['reward'],
                "task": results['task']
            })
        else:
            dataset = Dataset.from_dict({
                "prompt": results['prompt'],
                "completion": results['completion'],
                "answer": results['answer'],
                "reward": results['reward'],
            })
        return dataset