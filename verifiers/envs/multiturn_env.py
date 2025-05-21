from abc import abstractmethod
import asyncio
from asyncio import Semaphore
from copy import deepcopy
from typing import List, Dict, Any, Tuple

from datasets import Dataset
from openai import OpenAI

from verifiers.parsers import Parser
from verifiers.rubrics import Rubric
from verifiers.envs.environment import Environment


class MultiTurnEnv(Environment):
    def __init__(self,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 system_prompt: str | None = None,
                 few_shot: List[Dict[str, str]] | None = None,
                 parser: Parser = Parser(),
                 rubric: Rubric = Rubric(),
                 sampling_args: Dict[str, Any] = {},
                 max_concurrent: int = 32,
                 max_steps: int = 10,
                 **kwargs):
        super().__init__(
            client=client,
            model=model,
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=parser,
            rubric=rubric,
            **kwargs
        )
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        if dataset is not None:
            self.dataset = self.format_dataset(
                dataset=dataset,
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
        else:
            self.dataset = None
        if eval_dataset is not None:
            self.eval_dataset = self.format_dataset(
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
        if sampling_args is not None and "extra_body" in sampling_args:
            self.sampling_args["extra_body"].update(sampling_args["extra_body"])
        for k, v in sampling_args.items():
            if k != "extra_body":
                self.sampling_args[k] = v
        self.max_concurrent = max_concurrent
        self.max_steps = max_steps

    def get_dataset(self, n: int = -1, seed: int = 0, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.dataset is not None:
            return self.dataset.shuffle(seed=seed).select(range(n)) # type: ignore
        return self.dataset

    def get_eval_dataset(self, n: int = -1, seed: int = 0, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.eval_dataset is not None:
            return self.eval_dataset.shuffle(seed=seed).select(range(n)) # type: ignore
        return self.eval_dataset

    def format_prompt(self,
                      prompt: str,
                      system_prompt: str | None = None,
                      few_shot: List[Dict[str, str]] | None = None
                      ) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if few_shot:
            messages.extend(few_shot)
        messages.append({"role": "user", "content": prompt})
        return messages

    def format_dataset(self,
                       dataset: Dataset,
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, str]] | None = None,
                       question_key: str = "question",
                       answer_key: str = "answer") -> Dataset:
        if answer_key == "answer":
            return dataset.map(lambda x: {
                "prompt": self.format_prompt(x[question_key], system_prompt, few_shot),
            }, num_proc=self.max_concurrent)
        else:
            return dataset.map(lambda x: {
                "prompt": self.format_prompt(x[question_key], system_prompt, few_shot),
                "answer": x[answer_key]
            }, num_proc=self.max_concurrent)

    @abstractmethod
    def is_completed(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        pass

    def flatten(self, results: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Flatten a list of dictionaries into a single dictionary with lists of values.
        """
        return {k: [item[k] for item in results] for k in results[0]}

    def expand(self, results: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Expand a dictionary with lists of values into a list of dictionaries.
        """
        n = len(next(iter(results.values())))
        return [{k: v[i] for k, v in results.items()} for i in range(n)]

    def step(self,
             client: OpenAI,
             model: str,
             messages: List[Dict[str, str]],
             state: Dict[str, Any],
             sampling_args: Dict[str, Any] = {},
             **kwargs: Any) -> Tuple[List[Dict[str, str]], Dict[str, Any], bool]:
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
        msgs = deepcopy(messages)
        current_state = deepcopy(state)
        try:            
            # Get assistant response
            response = client.chat.completions.create(
                model=model,
                messages=msgs, # type: ignore
                **sampling_args
            )

            # Add assistant response to messages
            response_msg = {
                "role": "assistant", 
                "content": response.choices[0].message.content
            }
            msgs.append(response_msg)
            
            # Check if we're done
            if self.is_completed(msgs, current_state, **kwargs):
                is_completed = True
                next_state = current_state
            else:
                is_completed = False
                # If not done, get and add environment response
                env_msg, next_state = self.env_response(msgs, current_state, **kwargs)
                msgs.append(env_msg)
            return msgs, next_state, is_completed
            
        except Exception as e:
            # Handle errors by adding error message and returning
            error_msg = {"role": "assistant", "content": f"Error: {str(e)}"}
            msgs.append(error_msg)
            return msgs, current_state, True

    def generate(self,
                 inputs: List[Dict[str, Any]] | Dataset,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 sampling_args: Dict[str, Any] = {},
                 max_concurrent: int = 32,
                 score_rollouts: bool = True,
                 **kwargs: Any) -> Dict[str, List[Any]]:
        if client is None:
            assert self.client is not None
            client = self.client
        if model is None:
            assert self.model is not None
            model = self.model

        gen_sampling_args = deepcopy(self.sampling_args)
        gen_sampling_args.update(sampling_args)
        """
        Generate rollouts for a set of inputs.
        """
        def run_generate():
            # Get the evaluation dataset
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
                                lambda: self.step(
                                    client=client,
                                    model=model,
                                    messages=messages,
                                    state=state,
                                    sampling_args=gen_sampling_args,
                                    **kwargs
                                )
                            ) 
                            # Unpack the step_api result
                            messages, state, is_completed = step_result
                            
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
                tasks = [process_example(example, semaphore) for example in inputs]
                results = await tqdm_asyncio.gather(
                    *tasks,
                    total=len(inputs),
                    desc=f"Evaluating {len(inputs)} examples"
                )
                return results

            # Run generation (async)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_all_examples())
            finally:
                loop.close()

            results = self.flatten(results)
            if score_rollouts:
                # Score rollouts (async)
                results_rewards = self.rubric.score_rollout_group( 
                    prompts=results['prompt'],
                    completions=results['completion'],
                    answers=results['answer'],
                    tasks=results['task'],
                    max_concurrent=max_concurrent,
                    apply_weights=True
                )       
                results_rewards = self.flatten(results_rewards)
                results.update(results_rewards)
            return results
        return run_generate()

    def evaluate(self,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 sampling_args: Dict[str, Any] = {},
                 num_samples: int = -1,
                 max_concurrent: int = 32,
                 **kwargs: Any
                ) -> Dict[str, Any]:
        # use class-level client and model if not provided
        if client is None:
            assert self.client is not None
            client = self.client
        if model is None:
            assert self.model is not None
            model = self.model

        if self.eval_dataset is None:
            self.logger.info("eval_dataset is not set, falling back to train dataset")
            assert self.dataset is not None
            num_rows = len(self.dataset)
            inputs = self.dataset.select(range(num_rows - num_samples, num_rows))
        else:
            num_rows = len(self.eval_dataset)
            inputs = self.eval_dataset

        results = self.generate(
            inputs, client, model, sampling_args, **kwargs
        )
        return results

    def make_dataset(self,
                     results: Dict[str, Any] | None = None,
                     push_to_hub: bool = False,
                     hub_name: str | None = None,
                     client: OpenAI | None = None,
                     model: str | None = None,
                     max_concurrent: int | None = None,
                     num_samples: int = -1,
                     sampling_args: Dict[str, Any] = {"temperature": 0.6},
                     **kwargs: Any) -> Dataset:
        """
        Make a dataset from the evaluation results.
        """
        if results is None and client is None:
            raise ValueError("Either results or client must be provided")
        if push_to_hub and hub_name is None:
            raise ValueError("hub_name must be provided if push_to_hub is True")
        
        # use class-level client and model if not provided
        if client is None:
            assert self.client is not None
            client = self.client
        if model is None:
            assert self.model is not None
            model = self.model
        if max_concurrent is None:
            max_concurrent = self.max_concurrent
        if results is None:
            results = self.evaluate(
                client,
                model, 
                sampling_args,
                max_concurrent, 
                num_samples, 
                **kwargs
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
        if push_to_hub:
            assert hub_name is not None
            dataset.push_to_hub(hub_name)
        return dataset