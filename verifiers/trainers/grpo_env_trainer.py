# adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

import os
import logging
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Callable, Optional, Union, Any, List, Dict, Tuple

import datasets
import openai
import torch
from torch.utils.data import DataLoader, Sampler
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from peft import PeftConfig, get_peft_model, PeftModel, PeftMixedModel
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import seed_worker
from trl.models import create_reference_model, prepare_deepspeed
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import (
    disable_dropout_in_model,
    pad,
    selective_log_softmax
)
import wandb
import time

from verifiers import Environment
from verifiers.inference import VLLMClient
from verifiers.trainers.grpo_env_config import GRPOEnvConfig
from verifiers.trainers.async_batch_generator import AsyncBatchGenerator, BatchRequest, BatchResult
from verifiers.trainers.async_dataloader_wrapper import AsyncDataLoaderWrapper
from verifiers.utils.logging_utils import print_prompt_completions_sample   
from verifiers.utils.trainer_utils import RepeatSampler

# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]

def shuffle_tensor_dict(tensor_dict: dict[str, Optional[torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.

    Example:
        >>> x = torch.arange(6).reshape(3, 2)
        >>> y = torch.arange(3).reshape(3, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> shuffle_tensor_dict(tensor_dict)
        {'x': tensor([[2, 3],
                      [0, 1],
                      [4, 5]]),
         'y': tensor([[1],
                      [0],
                      [2]])}
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])

def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class GRPOEnvTrainer(Trainer):
    def __init__(
            self,
            model: PreTrainedModel,
            env: Environment,
            args: GRPOEnvConfig,
            processing_class: PreTrainedTokenizerBase,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional[PeftConfig] = None,
            **kwargs,
    ): 
        self.logger = logging.getLogger(__name__)
        
        # Models
        if peft_config is not None:
            model = get_peft_model(model, peft_config) # type: ignore

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args) # type: ignore
        
        # Suppress irrelevant warning
        model.warnings_issued["estimate_tokens"] = True 

        # Tokenizer pad token
        if processing_class.pad_token is None: # type: ignore
            processing_class.pad_token = processing_class.eos_token # type: ignore

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.max_concurrent = args.max_concurrent
        self.max_num_processes = args.max_num_processes
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.presence_penalty = args.presence_penalty
        self.frequency_penalty = args.frequency_penalty
        self.top_k = args.top_k
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions
        self.delta = args.delta

        # Reference model parameters
        self.beta = args.beta
        self.sync_ref_model = args.sync_ref_model
        self.generation_batch_size: int = args.generation_batch_size # type: ignore

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self._step = 0
        self._buffered_inputs = None

        # Data 
        self.shuffle_dataset = args.shuffle_dataset 
        train_dataset = env.get_dataset()
        assert train_dataset is not None
        
        # Filter out prompts that are too long if max_prompt_length is set
        if self.max_prompt_length is not None:
            self.logger.info(f"Filtering dataset for prompts with length <= {self.max_prompt_length}")
            max_length = self.max_prompt_length  # Capture for closure
            
            def filter_by_prompt_length(example):
                prompt = example['prompt']
                # Tokenize prompt to check length
                if isinstance(prompt, list):
                    # Chat format
                    prompt_text = processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                else:
                    # Completion format
                    prompt_text = prompt
                prompt_ids = processing_class.encode(prompt_text)
                return len(prompt_ids) <= max_length
            
            original_size = len(train_dataset)
            train_dataset = train_dataset.filter(filter_by_prompt_length, num_proc=self.max_num_processes)
            filtered_size = len(train_dataset)
            if filtered_size < original_size:
                self.logger.info(f"Filtered dataset from {original_size} to {filtered_size} examples ({original_size - filtered_size} prompts were too long)")
        
        def data_collator(features):
            return features
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Reference model
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            model_id = model.config._name_or_path
            model_init_kwargs = {"torch_dtype": "auto"}
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model) # type: ignore

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print

        # Environment integration parameters
        self.mask_env_responses = getattr(args, 'mask_env_responses', False)
        self.env_max_concurrent = getattr(args, 'env_max_concurrent', 32)

        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }
 
        # OpenAI client for Environment generation (using vLLM server)
        host = args.vllm_server_host
        port = args.vllm_server_port
        vllm_base_url = f"http://{host}:{port}/v1"
        self.oai_client = openai.OpenAI(base_url=vllm_base_url, api_key="EMPTY")

        # vLLM client for weight syncing only
        self.vllm_client = VLLMClient(
            host=host,
            port=port,
            connection_timeout=args.vllm_server_timeout
        )
        # Only initialize communicator on the main process
        # Other processes will only use the client for non-NCCL operations
        if self.accelerator.is_main_process:
            self.vllm_client.init_communicator()
        
        self._last_loaded_step = 0  # Initialize to 0 since vLLM already has initial weights
        self.model_accepts_loss_kwargs = False 
        # Weight updates to vLLM happen only when generating new completions
        # Frequency: every (gradient_accumulation_steps * num_iterations) training steps
        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

        # Reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        if self.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator)) 

        # Environment
        self.env = env

        # Async generation setup - always use async, num_steps_async=0 means synchronous behavior
        self._next_batch_id = 0
        self._async_started = False
        self.num_steps_async = args.num_steps_async
        
        # Always create async generator
        # num_steps_async=0 will behave synchronously (submit and wait immediately)
        self.async_generator = AsyncBatchGenerator(
            env=self.env,
            client=self.oai_client,
            model_name=self._get_model_name(),
            sampling_args=self._get_sampling_args(),
            num_steps_ahead=self.num_steps_async,
            max_queue_size=args.async_max_queue_size,
            generation_timeout=args.async_generation_timeout,
        )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        batch_size = self._train_batch_size * self.gradient_accumulation_steps

        dataloader_params = {
            "batch_size": batch_size, # type: ignore (None case handled by config __post_init__)
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(train_dataset, **dataloader_params)
        
        # Always wrap with AsyncDataLoaderWrapper for consistent behavior
        # Store the wrapped dataloader for async access
        self._async_dataloader = AsyncDataLoaderWrapper(
            dataloader, 
            buffer_size=max(5, self.num_steps_async * 2)
        )
        return self.accelerator.prepare(self._async_dataloader)

    def _get_train_sampler(self, train_dataset = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |    Accum step 0     |
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first gradient_accumulation_steps (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  grad_accum=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second gradient_accumulation_steps (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...

        return RepeatSampler(
            data_source=self.train_dataset, # type: ignore
            mini_repeat_count=self.num_generations,
            batch_size=self.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOEnvConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable() # type: ignore
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        assert isinstance(gradient_checkpointing_kwargs, dict)
        use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", True)

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed
            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext
            
        # Ensure all processes are synchronized before weight update
        self.accelerator.wait_for_everyone()

        # ALL processes must participate in model operations for DeepSpeed ZeRO-3
        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging
            with gather_if_zero3(list(self.model.parameters())):  # type: ignore
                self.model.merge_adapter() # type: ignore
                
                # Update vLLM weights while parameters are gathered
                # Create a list to collect all parameters on all processes
                all_params = []
                for name, param in self.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")
                    all_params.append((name, param.data))
                
                # Only main process sends to vLLM
                if self.accelerator.is_main_process:
                    for name, param_data in all_params:
                        self.vllm_client.update_named_param(name, param_data)
                    
                self.model.unmerge_adapter() # type: ignore
        else:
            # For non-PEFT models, gather and update each parameter individually
            # ALL processes must iterate through parameters
            param_count = 0
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                        param_count += 1
            
        # Reset cache on vLLM (main process only)
        if self.accelerator.is_main_process:
            self.logger.info(f"Resetting vLLM prefix cache")
            self.vllm_client.reset_prefix_cache()
        
        # Ensure all processes wait for the main process to finish updating weights
        self.accelerator.wait_for_everyone()

    def _prepare_inputs(
        self, inputs: Union[dict[str, Union[torch.Tensor, Any]], list[dict[str, Union[torch.Tensor, Any]]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # Always uses async generation (num_steps_async=0 behaves synchronously)
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × gradient accumulation steps)
        #   - Generates completions once for the entire generation batch and splits it into batches
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every gradient_accumulation_steps * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        # Handle both dict and list inputs
        generation_batch = inputs if isinstance(inputs, list) else [inputs]
        
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.gradient_accumulation_steps * self.num_iterations
            
            # Prime the async pipeline on first use (dummy round)
            if self._step == 0 and self.async_generator is not None and self.async_generator.num_steps_ahead > 0:
                # All processes must participate in priming to avoid deadlock
                num_batches_to_prime = self.async_generator.num_steps_ahead
                
                if self.accelerator.is_main_process:
                    self.logger.info(f"Priming async pipeline with {num_batches_to_prime} batches")
                    
                    # Start the async generator
                    self.async_generator.start()
                    self._async_started = True
                
                # Submit initial batches (all processes participate in gather operations)
                for i in range(num_batches_to_prime):
                    # Main process peeks at future batches
                    future_batch = None
                    batch_exists = False
                    
                    if self.accelerator.is_main_process:
                        future_batches = self._async_dataloader.peek_ahead(i + 1)
                        if future_batches and len(future_batches) > i:
                            future_batch = future_batches[i]
                            if isinstance(future_batch, dict):
                                future_batch = [future_batch]
                            batch_exists = True
                    
                    # Broadcast whether batch exists to all processes
                    batch_exists_list = [batch_exists]
                    broadcast_object_list(batch_exists_list, from_process=0)
                    batch_exists = batch_exists_list[0]
                    
                    if not batch_exists:
                        break  # No more batches available
                    
                    # ALL processes must participate in gather
                    if self.accelerator.is_main_process and future_batch is not None:
                        prompts = [x['prompt'] for x in future_batch]
                        answers = [x['answer'] for x in future_batch]
                        tasks = [x.get('task', 'default') for x in future_batch]
                    else:
                        prompts = []
                        answers = []
                        tasks = []
                    
                    all_prompts = gather_object(prompts)
                    all_answers = gather_object(answers)
                    all_tasks = gather_object(tasks)
                    
                    # Only main process submits
                    if self.accelerator.is_main_process:
                        request = BatchRequest(
                            batch_id=i,
                            env_inputs={'prompt': all_prompts, 'answer': all_answers, 'task': all_tasks},
                            processing_class=self.processing_class,
                            mask_env_responses=self.mask_env_responses,
                            max_concurrent=self.max_concurrent,
                            device=self.accelerator.device,
                            accelerator=self.accelerator,
                            process_index=self.accelerator.process_index,
                            num_processes=self.accelerator.num_processes,
                            local_batch_size=len(future_batch),
                        )
                        self.async_generator.submit_batch(request)
                        self._next_batch_id = i + 1
                
                if self.accelerator.is_main_process:
                    self.logger.info(f"Submitted {self._next_batch_id} batches to prime the pipeline")
                
                # All processes wait for priming to complete
                self.accelerator.wait_for_everyone()
            
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # Update weights to vLLM before generating new completions
                # Only update if we've made progress since last generation
                if self.state.global_step > self._last_loaded_step:
                    self.logger.info(f"Syncing weights to vLLM at step {self.state.global_step} before generation")
                    self._move_model_to_vllm()
                    self._last_loaded_step = self.state.global_step
                    self.logger.info(f"Weight sync complete")
                
                # Always use async generation path
                processed_batch = self._handle_async_generation(generation_batch)
                processed_batch = shuffle_tensor_dict(processed_batch)
                self._buffered_inputs = split_tensor_dict(processed_batch, self.gradient_accumulation_steps)
            result = self._buffered_inputs[self._step % self.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, generate without buffering
            # Still use async path for consistency
            result = self._handle_async_generation(generation_batch)
        return result

    def _handle_async_generation(self, generation_batch: list[dict[str, Union[torch.Tensor, Any]]]) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Handle async generation with aggressive consolidation:
        - PRIMARY process does ALL processing (generation, token processing, advantages)
        - NON-PRIMARY processes just contribute data and wait for results
        """
        device = self.accelerator.device
        
        # Step 1: Gather batch inputs from all processes
        all_prompts, all_answers, all_tasks = self._gather_batch_inputs(generation_batch)
        
        # From here, only PRIMARY process does work, others wait
        if self.accelerator.is_main_process:
            # Initialize async generator on first use (if not already done in priming)
            if not self._async_started and self.async_generator is not None:
                self.async_generator.start()
                self._async_started = True
            
            # Determine current batch ID to retrieve
            # We generate once every (gradient_accumulation_steps * num_iterations) steps
            # So batch 0 is for steps 0-3, batch 1 is for steps 4-7, etc.
            batch_id_to_retrieve = self._step // (self.gradient_accumulation_steps * self.num_iterations)
            
            # Determine if we need to submit more batches
            if self.async_generator is not None:
                pending_count = self.async_generator.get_pending_count()
                completed_count = self.async_generator.get_completed_count()
                total_in_pipeline = pending_count + completed_count
                
                # Calculate how many batches ahead we should be
                # We want to stay num_steps_ahead batches ahead of what we're retrieving
                target_batch_id = batch_id_to_retrieve + self.async_generator.num_steps_ahead
                
                # Submit batches to reach our target
                while self._next_batch_id <= target_batch_id:
                    # Check if we have data for this batch
                    batch_offset = self._next_batch_id - batch_id_to_retrieve
                    if batch_offset < 0:
                        # This batch should have been pre-submitted during priming
                        self._next_batch_id += 1
                        continue
                        
                    # For current batch (offset 0), use the provided data
                    if batch_offset == 0:
                        request = BatchRequest(
                            batch_id=self._next_batch_id,
                            env_inputs={'prompt': all_prompts, 'answer': all_answers, 'task': all_tasks},
                            processing_class=self.processing_class,
                            mask_env_responses=self.mask_env_responses,
                            max_concurrent=self.max_concurrent,
                            device=device,
                            accelerator=self.accelerator,
                            process_index=self.accelerator.process_index,
                            num_processes=self.accelerator.num_processes,
                            local_batch_size=len(generation_batch),
                        )
                        self.async_generator.submit_batch(request)
                        self._next_batch_id += 1
                    else:
                        # For future batches, peek ahead in the dataloader
                        # ALL processes must participate in gather operations
                        future_batch_data = None
                        has_future_batch = False
                        
                        if self.accelerator.is_main_process:
                            if hasattr(self, '_async_dataloader'):
                                future_batches = self._async_dataloader.peek_ahead(batch_offset)
                                if future_batches and len(future_batches) > batch_offset - 1:
                                    future_batch = future_batches[batch_offset - 1]
                                    if isinstance(future_batch, dict):
                                        future_batch = [future_batch]
                                    has_future_batch = True
                                    future_batch_data = {
                                        'prompts': [x['prompt'] for x in future_batch],
                                        'answers': [x['answer'] for x in future_batch],
                                        'tasks': [x.get('task', 'default') for x in future_batch],
                                        'batch_size': len(future_batch)
                                    }
                        
                        # Broadcast whether we have a future batch to all processes
                        has_future_batch_list = [has_future_batch]
                        broadcast_object_list(has_future_batch_list, from_process=0)
                        has_future_batch = has_future_batch_list[0]
                        
                        if not has_future_batch:
                            break  # No more data available
                        
                        # Broadcast future batch data
                        future_batch_data_list = [future_batch_data]
                        broadcast_object_list(future_batch_data_list, from_process=0)
                        future_batch_data = future_batch_data_list[0]
                        
                        # ALL processes participate in gather
                        if self.accelerator.is_main_process:
                            prompts = future_batch_data['prompts']
                            answers = future_batch_data['answers']
                            tasks = future_batch_data['tasks']
                        else:
                            prompts = []
                            answers = []
                            tasks = []
                        
                        all_future_prompts = gather_object(prompts)
                        all_future_answers = gather_object(answers)
                        all_future_tasks = gather_object(tasks)
                        
                        # Only main process submits
                        if self.accelerator.is_main_process:
                            future_request = BatchRequest(
                                batch_id=self._next_batch_id,
                                env_inputs={'prompt': all_future_prompts, 'answer': all_future_answers, 'task': all_future_tasks},
                                processing_class=self.processing_class,
                                mask_env_responses=self.mask_env_responses,
                                max_concurrent=self.max_concurrent,
                                device=device,
                                accelerator=self.accelerator,
                                process_index=self.accelerator.process_index,
                                num_processes=self.accelerator.num_processes,
                                local_batch_size=future_batch_data['batch_size'],
                            )
                            self.async_generator.submit_batch(future_request)
                            self._next_batch_id += 1
                
                # Get the batch result we need
                try:
                    batch_result = self.async_generator.get_batch(batch_id_to_retrieve)
                    if batch_result.error:
                        raise batch_result.error
                except Exception as e:
                    raise RuntimeError(f"Async generation failed for batch {batch_id_to_retrieve}: {e}")
            else:
                raise RuntimeError("Async generator not initialized on main process")
            
            # Process everything on primary
            processed_results = batch_result.processed_results
            
            # Convert rewards and compute advantages using FULL batch
            all_rewards = processed_results['rewards']
            all_rewards = torch.tensor(all_rewards, device=device) if not isinstance(all_rewards, torch.Tensor) else all_rewards
            all_advantages = self._compute_advantages(all_rewards)
            
            # Process all token sequences
            all_prompt_ids = []
            all_prompt_mask = []
            all_completion_ids = []
            all_completion_mask = []
            
            for i in range(len(processed_results['prompt_ids'])):
                prompt_ids = torch.tensor(processed_results['prompt_ids'][i], device=device)
                prompt_mask = torch.tensor(processed_results['prompt_mask'][i], device=device)
                completion_ids = torch.tensor(processed_results['completion_ids'][i], device=device)
                completion_mask = torch.tensor(processed_results['completion_mask'][i], device=device)
                
                all_prompt_ids.append(prompt_ids)
                all_prompt_mask.append(prompt_mask)
                all_completion_ids.append(completion_ids)
                all_completion_mask.append(completion_mask)
            
            # Pad all sequences
            all_prompt_ids = pad(all_prompt_ids, padding_value=self.processing_class.pad_token_id, padding_side='left')
            all_prompt_mask = pad(all_prompt_mask, padding_side='left')
            all_completion_ids = pad(all_completion_ids, padding_value=self.processing_class.pad_token_id, padding_side='right')
            all_completion_mask = pad(all_completion_mask)
            
            # Truncate if needed
            if self.max_prompt_length is not None and all_prompt_ids.size(1) > self.max_prompt_length:
                all_prompt_ids = all_prompt_ids[:, -self.max_prompt_length:]
                all_prompt_mask = all_prompt_mask[:, -self.max_prompt_length:]
            
            if self.max_completion_length is not None and all_completion_ids.size(1) > self.max_completion_length:
                all_completion_ids = all_completion_ids[:, :self.max_completion_length]
                all_completion_mask = all_completion_mask[:, :self.max_completion_length]
            
            # Log metrics and textual data
            all_reward_dict = batch_result.all_reward_dict if hasattr(batch_result, 'all_reward_dict') else {'reward': processed_results['rewards']}
            all_completions = batch_result.completions if hasattr(batch_result, 'completions') else []
            
            self._log_generation_metrics_primary(
                mode="train",
                all_reward_dict=all_reward_dict,
                all_rewards=all_rewards,
                generation_batch_size=len(generation_batch) * self.accelerator.num_processes
            )
            
            self._log_textual_data_primary(
                all_prompts=all_prompts,
                all_completions=all_completions,
                all_reward_dict=all_reward_dict
            )
            
            # Package everything for broadcast
            broadcast_data = {
                'prompt_ids': all_prompt_ids,
                'prompt_mask': all_prompt_mask,
                'completion_ids': all_completion_ids,
                'completion_mask': all_completion_mask,
                'advantages': all_advantages,
                'rewards': all_rewards,
            }
        else:
            # Non-primary processes just wait
            broadcast_data = None
        
        # Step 2: Broadcast all processed data to other processes
        broadcast_list = [broadcast_data]
        broadcast_object_list(broadcast_list, from_process=0)
        broadcast_data = broadcast_list[0]
        
        # Step 3: Each process takes its slice
        process_slice = slice(
            self.accelerator.process_index * len(generation_batch),
            (self.accelerator.process_index + 1) * len(generation_batch),
        )
        
        prompt_ids = broadcast_data['prompt_ids'][process_slice]
        prompt_mask = broadcast_data['prompt_mask'][process_slice]
        completion_ids = broadcast_data['completion_ids'][process_slice]
        completion_mask = broadcast_data['completion_mask'][process_slice]
        advantages = broadcast_data['advantages'][process_slice]
        
        # Store for local metrics if needed
        self._current_prompt_mask = prompt_mask
        
        # Log local completion metrics
        self._log_local_completion_metrics(
            mode="train",
            completion_mask=completion_mask,
            completion_ids=completion_ids,
            prompt_mask=prompt_mask
        )
        
        # Note: Don't increment _step here, that happens in _prepare_inputs
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": None,
            "advantages": advantages,
        }

    # ========== Helper methods for code consolidation ==========

    def _gather_batch_inputs(self, inputs: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """Gather prompts, answers, and tasks from all processes."""
        prompts = [x['prompt'] for x in inputs]
        answers = [x['answer'] for x in inputs]
        tasks = [x.get('task', 'default') for x in inputs]
        
        all_prompts = gather_object(prompts)
        all_answers = gather_object(answers)
        all_tasks = gather_object(tasks)
        
        return all_prompts, all_answers, all_tasks

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantages from rewards with normalization using full batch statistics."""
        # Always use full batch statistics
        mean_grouped = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute advantages
        mean_grouped = mean_grouped.repeat_interleave(self.num_generations, dim=0)
        std_grouped = std_grouped.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped
        
        if self.scale_rewards:
            advantages = advantages / (std_grouped + 1e-4)
        
        return advantages

    # ========== End of helper methods ==========

    def _log_generation_metrics_primary(
        self,
        mode: str,
        all_reward_dict: Dict[str, Any],
        all_rewards: torch.Tensor,
        generation_batch_size: int
    ) -> None:
        """
        Log generation metrics (PRIMARY PROCESS ONLY).
        This handles reward statistics and per-reward-function metrics using the full batch data.
        """
        # Log reward statistics using full batch
        mean_rewards = all_rewards.view(-1, self.num_generations).mean(dim=1)
        std_rewards = all_rewards.view(-1, self.num_generations).std(dim=1)
        self._metrics[mode]["reward"].append(mean_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        
        # Log individual reward function scores as metrics
        for reward_key in all_reward_dict:
            if reward_key != 'reward':  # Skip the consolidated reward
                reward_values = all_reward_dict[reward_key]
                if isinstance(reward_values, list):
                    reward_tensor = torch.tensor(reward_values, device=all_rewards.device)
                else:
                    reward_tensor = reward_values
                mean_reward = reward_tensor.mean().item()
                self._metrics[mode][f"rewards/{reward_key}"].append(mean_reward)

    def _log_textual_data_primary(
        self,
        all_prompts: List[str],
        all_completions: List[str],
        all_reward_dict: Dict[str, Any]
    ) -> None:
        """
        Log textual data for wandb (PRIMARY PROCESS ONLY).
        This logs the full batch of prompts, completions, and rewards.
        """
        self._textual_logs["prompt"].extend(all_prompts)
        self._textual_logs["completion"].extend(all_completions)
        
        # Log all reward scores - both individual functions and consolidated
        for reward_key in all_reward_dict:
            reward_values = all_reward_dict[reward_key]
            self._textual_logs["rewards"][reward_key].extend(
                reward_values.tolist() if isinstance(reward_values, torch.Tensor) else reward_values
            )

    def _log_local_completion_metrics(
        self,
        mode: str,
        completion_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_mask: torch.Tensor
    ) -> None:
        """
        Log completion-related metrics (ALL PROCESSES).
        These metrics need local data and will be gathered later by accelerator.gather_for_metrics.
        """
        # Log token count
        if mode == "train":
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        
        # Log completion lengths - gather_for_metrics will aggregate across processes
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())
        
        # Check for EOS tokens and log terminated sequence lengths
        is_eos = completion_ids == self.processing_class.eos_token_id
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=completion_mask.device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

    def _get_sampling_args(self) -> Dict[str, Any]:
        """Get sampling arguments for Environment generation."""
        args = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_completion_length,
            'n': 1,
            'presence_penalty': self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
            'extra_body': {
                'top_k': self.top_k,
                'min_p': self.min_p,
                'repetition_penalty': self.repetition_penalty,
            }
        }
        return args

    def _get_model_name(self) -> str:
        """Get model name for Environment generation."""
        return self.model.config._name_or_path # type: ignore

    def _ids_to_tensors(self,
                        prompt_ids: List[List[int]],
                        prompt_mask: List[List[int]],
                        completion_ids: List[List[int]],
                        completion_mask: List[List[int]],
                        device: torch.device) -> Dict[str, torch.Tensor]:
    
        ids = [prompt_ids[i] + completion_ids[i] for i in range(len(prompt_ids))] 
        mask = [prompt_mask[i] + completion_mask[i] for i in range(len(prompt_mask))]
        max_len = max(len(ids[i]) for i in range(len(ids)))
        ids = [torch.cat([
            torch.tensor(ids[i], dtype=torch.long, device=device),
            torch.zeros(max_len - len(ids[i]), dtype=torch.long, device=device)
        ]) for i in range(len(ids))]
        mask = [torch.cat([
            torch.tensor(mask[i], dtype=torch.long, device=device),
            torch.zeros(max_len - len(mask[i]), dtype=torch.long, device=device)
        ]) for i in range(len(mask))]
        ids = torch.stack(ids, dim=0)   
        mask = torch.stack(mask, dim=0)
        return {
            'ids': ids,
            'mask': mask
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs") 
        mode = "train" if self.model.training else "eval" # type: ignore
        
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps,
        # so we can skip it's computation (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.delta is not None:
            # Use clamp instead of min to handle tensor-float comparison
            per_token_loss1 = torch.clamp(coef_1, max=self.delta) * advantages.unsqueeze(1)
        else:
            # Original GRPO clipping (only lower bound implicitly applied by the final min)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)

        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter(): # type: ignore
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item()) # type: ignore

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item()) # type: ignore
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item()) # type: ignore
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item()) # type: ignore
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item()) # type: ignore
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item()) # type: ignore
        return loss
 
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model is not None and self.model.training else "eval" # type: ignore
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if len(self._textual_logs["prompt"]) > 0:
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd
 
                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})
            
            # Clear the textual logs after logging
            self._textual_logs["prompt"].clear()
            self._textual_logs["completion"].clear()
            for key in self._textual_logs["rewards"]:
                self._textual_logs["rewards"][key].clear()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to ensure async generation is properly managed"""
        # Weight updates now happen in _prepare_inputs right before generation
        # Continue with normal training
        return super().training_step(model, inputs, num_items_in_batch)
    
    def _inner_training_loop(self, *args, **kwargs):
        """Override to ensure async generator is stopped when training ends"""
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            # Clean up async generator on all processes
            if self.async_generator and self._async_started and self.accelerator.is_main_process:
                self.async_generator.stop()
            self._async_started = False