# adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

import os
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
        self.steps_per_generation: int = args.steps_per_generation # type: ignore

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self._step = 0
        self._buffered_inputs = None

        # Data 
        self.shuffle_dataset = args.shuffle_dataset 
        train_dataset = env.get_dataset()
        assert train_dataset is not None
        
        # Filter out prompts that are too long if max_prompt_length is set
        if self.max_prompt_length is not None:
            print(f"Filtering dataset for prompts with length <= {self.max_prompt_length}")
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
                print(f"Filtered dataset from {original_size} to {filtered_size} examples ({original_size - filtered_size} prompts were too long)")
        
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
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation * args.gradient_accumulation_steps
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
            # Log NCCL environment variables for debugging
            nccl_vars = {k: v for k, v in os.environ.items() if k.startswith('NCCL')}
            if nccl_vars:
                print(f"[TRAINER] NCCL environment variables: {nccl_vars}")
            self.vllm_client.init_communicator()
        
        self._last_loaded_step = 0  # Initialize to 0 since vLLM already has initial weights
        self.model_accepts_loss_kwargs = False 
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
            assert isinstance(self.ref_model, PreTrainedModel)
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator)) 

        # Environment
        self.env = env

        # Async generation setup
        self.use_async_generation = args.use_async_generation
        self.async_generator = None
        self._next_batch_id = 0
        self._async_started = False
        
        if self.use_async_generation:
            self.async_generator = AsyncBatchGenerator(
                env=self.env,
                client=self.oai_client,
                model_name=self._get_model_name(),
                sampling_args=self._get_sampling_args(),
                num_steps_ahead=args.num_steps_async,
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

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation, # type: ignore (None case handled by config __post_init__)
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
        
        # Wrap with AsyncDataLoaderWrapper if using async generation
        if self.use_async_generation:
            # Store the wrapped dataloader for async access
            self._async_dataloader = AsyncDataLoaderWrapper(
                dataloader, 
                buffer_size=max(5, self.args.num_steps_async * 2)
            )
            return self.accelerator.prepare(self._async_dataloader)
        else:
            return self.accelerator.prepare(dataloader)

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
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...

        return RepeatSampler(
            data_source=self.train_dataset, # type: ignore
            mini_repeat_count=self.num_generations,
            batch_size=self.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.steps_per_generation,
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

        # Only the main process updates weights to vLLM
        # This is because only the main process has initialized the NCCL communicator
        if not self.accelerator.is_main_process:
            # Non-main processes wait here while main process updates weights
            self.accelerator.wait_for_everyone()
            return

        print(f"[TRAINER] Starting weight update to vLLM")
        update_start_time = time.time()
        
        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):  # type: ignore
                self.model.merge_adapter() # type: ignore
                
                # Create a temporary model wrapper for PEFT to handle parameter name mapping
                class PEFTModelWrapper:
                    def __init__(self, peft_model):
                        self.peft_model = peft_model
                    
                    def named_parameters(self):
                        for name, param in self.peft_model.named_parameters():
                            # When using PEFT, we need to recover the original parameter name and discard some parameters
                            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                            if self.peft_model.prefix in name:
                                continue
                            # When module to save, remove its prefix and discard the original module
                            if "original_module" in name:
                                continue
                            name = name.replace("modules_to_save.default.", "")
                            yield name, param
                
                # Use batch update for PEFT model
                wrapper = PEFTModelWrapper(self.model)
                self.vllm_client.batch_update_model_params(wrapper, batch_size=10)
                
                self.model.unmerge_adapter() # type: ignore
        else:
            # For non-PEFT models, use batch update directly
            # For DeepSpeed ZeRO-3, we need to gather parameters as we go
            if zero_stage_3:
                # Create a wrapper that gathers parameters on demand
                class ZeRO3ModelWrapper:
                    def __init__(self, model, gather_fn):
                        self.model = model
                        self.gather_fn = gather_fn
                        # Pre-gather all parameters to avoid issues with the batch update
                        self.gathered_params = []
                        for name, param in model.named_parameters():
                            with gather_fn([param]):
                                # Clone the parameter to keep it after the context manager exits
                                self.gathered_params.append((name, param.data.clone()))
                    
                    def named_parameters(self):
                        for name, param_data in self.gathered_params:
                            # Create a dummy parameter object for the batch update
                            dummy_param = torch.nn.Parameter(param_data)
                            yield name, dummy_param
                
                wrapper = ZeRO3ModelWrapper(self.model, gather_if_zero3)
                self.vllm_client.batch_update_model_params(wrapper, batch_size=10)
            else:
                # Regular model without ZeRO-3
                self.vllm_client.batch_update_model_params(self.model, batch_size=10)  # type: ignore

        # Reset cache on vLLM
        print(f"[TRAINER] Resetting vLLM prefix cache")
        self.vllm_client.reset_prefix_cache()
        
        update_duration = time.time() - update_start_time
        print(f"[TRAINER] Weight update complete in {update_duration:.1f}s")
        
        # Ensure all processes wait for the main process to finish updating weights
        self.accelerator.wait_for_everyone()

    def _prepare_inputs(
        self, inputs: Union[dict[str, Union[torch.Tensor, Any]], list[dict[str, Union[torch.Tensor, Any]]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        # Handle both dict and list inputs
        generation_batch = inputs if isinstance(inputs, list) else [inputs]
        
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                if self.use_async_generation:
                    # Async generation path
                    processed_batch = self._handle_async_generation(generation_batch)
                else:
                    # Sync generation path (original behavior)
                    processed_batch = self._generate_and_score_completions(generation_batch)
                    
                processed_batch = shuffle_tensor_dict(processed_batch)
                self._buffered_inputs = split_tensor_dict(processed_batch, self.steps_per_generation)
            result = self._buffered_inputs[self._step % self.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            result = self._generate_and_score_completions(generation_batch)
        return result

    def _handle_async_generation(self, generation_batch: list[dict[str, Union[torch.Tensor, Any]]]) -> dict[str, Union[torch.Tensor, Any]]:
        """Handle async generation for training"""
        # Start async generator if not started
        if not self._async_started and self.async_generator is not None:
            self.async_generator.start()
            self._async_started = True
            # Submit initial batches
            for _ in range(min(self.async_generator.num_steps_ahead, 3)):  # Pre-submit up to 3 batches
                self._submit_next_batch_async()
        
        # Get the current batch result
        batch_result = self.async_generator.get_batch(self._next_batch_id)
        if batch_result.error:
            raise RuntimeError(f"Async generation failed for batch {self._next_batch_id}: {batch_result.error}")
        
        # Process the result
        processed_batch = self._process_async_result(batch_result, generation_batch)
        
        # Increment batch ID and submit next batch
        self._next_batch_id += 1
        self._submit_next_batch_async()
        
        return processed_batch
    
    def _submit_next_batch_async(self):
        """Submit the next batch for async generation if needed"""
        if not self.async_generator or not self.async_generator.should_submit_more():
            return
            
        # Calculate which batch we need
        pending_count = self.async_generator.get_pending_count()
        completed_count = self.async_generator.get_completed_count()
        batch_offset = pending_count + completed_count
        
        # Get future batches from the async dataloader
        if hasattr(self, '_async_dataloader'):
            future_batches = self._async_dataloader.get_future_batches(batch_offset, 1)
            if not future_batches:
                return  # No more batches available
                
            future_batch = future_batches[0]
            
            # Extract prompts and answers from the batch
            prompts = [x['prompt'] for x in future_batch]
            answers = [x['answer'] for x in future_batch]
            tasks = [x.get('task', 'default') for x in future_batch]
            
            # Gather across all processes
            all_prompts = gather_object(prompts)
            all_answers = gather_object(answers)
            all_tasks = gather_object(tasks)
            
            # Only main process submits the batch
            if self.accelerator.is_main_process:
                batch_id = self._next_batch_id + batch_offset
                
                request = BatchRequest(
                    batch_id=batch_id,
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

    def _process_async_result(self, batch_result: BatchResult, generation_batch: list[dict[str, Union[torch.Tensor, Any]]]) -> dict[str, Union[torch.Tensor, Any]]:
        """Process async generation result and convert to expected format"""
        device = self.accelerator.device
        processed_results = batch_result.processed_results
        
        # Get all reward scores from the batch result
        all_reward_dict = batch_result.all_reward_dict if hasattr(batch_result, 'all_reward_dict') else {'reward': processed_results['rewards']}
        
        # Slice data for this process
        process_slice = slice(
            self.accelerator.process_index * len(generation_batch),
            (self.accelerator.process_index + 1) * len(generation_batch),
        )
        
        for key, value in processed_results.items():
            processed_results[key] = value[process_slice]
        
        # Convert lists to tensors, concatenate, and pad
        prompt_ids = [torch.tensor(prompt_ids, device=device)
                      for prompt_ids in processed_results['prompt_ids']]
        prompt_mask = [torch.tensor(prompt_mask, device=device)
                      for prompt_mask in processed_results['prompt_mask']]
        prompt_ids = pad(prompt_ids, padding_value=self.processing_class.pad_token_id, padding_side='left')
        prompt_mask = pad(prompt_mask, padding_side='left')
        
        # Truncate prompts from the left if they exceed max_prompt_length
        if self.max_prompt_length is not None and prompt_ids.size(1) > self.max_prompt_length:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        
        completion_ids = [torch.tensor(completion_ids, device=device)
                          for completion_ids in processed_results['completion_ids']]
        completion_mask = [torch.tensor(completion_mask, device=device)
                          for completion_mask in processed_results['completion_mask']]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id, padding_side='right')
        completion_mask = pad(completion_mask)
        
        # Truncate completions from the right if they exceed max_completion_length
        if self.max_completion_length is not None and completion_ids.size(1) > self.max_completion_length:
            completion_ids = completion_ids[:, :self.max_completion_length]
            completion_mask = completion_mask[:, :self.max_completion_length]
        
        rewards = processed_results['rewards']
        
        rewards = torch.tensor(rewards, device=device)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
            
        # Log reward statistics to metrics
        mode = "train"  # async generation is only used in training
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        
        # Log individual reward function scores as metrics
        for reward_key in all_reward_dict:
            if reward_key != 'reward':  # Skip the consolidated reward
                reward_values = all_reward_dict[reward_key][process_slice]
                if isinstance(reward_values, list):
                    reward_tensor = torch.tensor(reward_values, device=device)
                else:
                    reward_tensor = reward_values
                mean_reward = reward_tensor.mean().item()
                self._metrics[mode][f"rewards/{reward_key}"].append(mean_reward)
            
        # Log all reward scores - both individual functions and consolidated
        completions = batch_result.completions[process_slice] if hasattr(batch_result, 'completions') else []
        prompts = [x['prompt'] for x in generation_batch]
        self._textual_logs["prompt"].extend(gather_object(prompts))
        self._textual_logs["completion"].extend(gather_object(completions))
        for reward_key in all_reward_dict:
            # Slice rewards for this process
            reward_values = all_reward_dict[reward_key][process_slice]
            self._textual_logs["rewards"][reward_key].extend(gather_object(reward_values.tolist() if isinstance(reward_values, torch.Tensor) else reward_values))
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": None,
            "advantages": advantages,
        }

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

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]   
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Main generation method using Environment.
        
        Data Flow:
        1. Convert trainer inputs to Environment format
        2. Call Environment.generate() (happens on main process)
        3. Process results using process_environment_results() (main process only, creates tensors on GPU)
        4. Broadcast final processed tensors to all processes
        5. Slice data for each process
        6. Compute advantages from rewards
        
        This method handles all device placement and accelerator distribution.
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval" # type: ignore
        prompts = [x['prompt'] for x in inputs] # type: ignore
        answers = [x['answer'] for x in inputs] # type: ignore
        all_prompts = gather_object(prompts)
        all_answers = gather_object(answers)
        if 'task' in inputs[0].keys():
            tasks = [x['task'] for x in inputs] # type: ignore
            all_tasks = gather_object(tasks)
        else:
            all_tasks = ['default'] * len(all_prompts)

        # generate via Environment (main process only)
        object_keys = ['prompt_ids', 'prompt_mask', 'completion_ids', 'completion_mask', 'rewards']
        all_completions = []  # To store completions for logging
        all_reward_dict = {}  # To store all reward scores for logging
        if self.accelerator.is_main_process:
            env_inputs = {'prompt': all_prompts, 'answer': all_answers, 'task': all_tasks} 
            env_results = self.env.generate(
                env_inputs,
                client=self.oai_client,
                model=self._get_model_name(),
                sampling_args=self._get_sampling_args(),
                score_rollouts=True,
                max_concurrent=self.max_concurrent,
            ) # prompts, completions, states, rewards

            # Store completions for logging
            all_completions = env_results['completion']
            
            # Extract all reward-related keys (individual functions + consolidated 'reward')
            reward_keys = [k for k in env_results.keys() if k.endswith('_func') or k == 'reward']
            for key in reward_keys:
                all_reward_dict[key] = env_results[key]

            processed_results = self.env.process_env_results(
                env_results['prompt'],
                env_results['completion'],
                env_results['state'],
                env_results['reward'],  # Use consolidated reward for training
                processing_class=self.processing_class, # type: ignore
                mask_env_responses=self.mask_env_responses
            ) # prompt_ids, prompt_mask, completion_ids, completion_mask, rewards
        else:
            env_results = None
            processed_results = {}

        # Broadcast completions for logging
        completions_list = [all_completions] if self.accelerator.is_main_process else [None]
        broadcast_object_list(completions_list, from_process=0)
        all_completions = completions_list[0] if completions_list[0] is not None else []
        
        # Broadcast all reward scores for logging
        reward_dict_list = [all_reward_dict] if self.accelerator.is_main_process else [None]
        broadcast_object_list(reward_dict_list, from_process=0)
        all_reward_dict = reward_dict_list[0] if reward_dict_list[0] is not None else {}

        # Step 4: Broadcast final processed tensors to all processes
        for key in object_keys:
            if self.accelerator.is_main_process:
                object_list = [processed_results.get(key, None)]
            else:
                object_list = [None]
            broadcast_object_list(object_list, from_process=0)
            if object_list[0] is not None:
                processed_results[key] = object_list[0]
        
        # Compute advantages BEFORE process slicing to use the full reward batch
        all_rewards = torch.tensor(processed_results['rewards'], device=device)
        mean_grouped_rewards = all_rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = all_rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        all_advantages = all_rewards - mean_grouped_rewards
        if self.scale_rewards:
            all_advantages = all_advantages / (std_grouped_rewards + 1e-4)
        
        # Step 5: Slice data for this process
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        
        for key, value in processed_results.items():
            processed_results[key] = value[process_slice]
        
        # Slice advantages, completions, and rewards for this process
        advantages = all_advantages[process_slice]
        completions = all_completions[process_slice] if all_completions else []
        rewards = all_rewards[process_slice]

        # convert lists to tensors, concatenate, and pad
        prompt_ids = [torch.tensor(prompt_ids, device=device)
                      for prompt_ids in processed_results['prompt_ids']]
        prompt_mask = [torch.tensor(prompt_mask, device=device)
                      for prompt_mask in processed_results['prompt_mask']]
        prompt_ids = pad(prompt_ids, padding_value=self.processing_class.pad_token_id, padding_side='left') # type: ignore
        prompt_mask = pad(prompt_mask, padding_side='left')
        
        # Truncate prompts from the left if they exceed max_prompt_length
        if self.max_prompt_length is not None and prompt_ids.size(1) > self.max_prompt_length:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        
        completion_ids = [torch.tensor(completion_ids, device=device)
                          for completion_ids in processed_results['completion_ids']]
        completion_mask = [torch.tensor(completion_mask, device=device)
                          for completion_mask in processed_results['completion_mask']]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id, padding_side='right') # type: ignore
        completion_mask = pad(completion_mask)
        
        # Truncate completions from the right if they exceed max_completion_length
        if self.max_completion_length is not None and completion_ids.size(1) > self.max_completion_length:
            completion_ids = completion_ids[:, :self.max_completion_length]
            completion_mask = completion_mask[:, :self.max_completion_length]

        # Log 
        if mode == "train":
            # Update token count
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item() # type: ignore
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item()) # type: ignore
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item()) # type: ignore
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item()) # type: ignore

        # Check for EOS tokens and log terminated sequence lengths
        is_eos = completion_ids == self.processing_class.eos_token_id # type: ignore
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1)) # type: ignore
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos] # type: ignore
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # Log reward statistics
        mean_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        self._metrics[mode]["reward"].append(mean_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        
        # Log individual reward function scores as metrics
        for reward_key in all_reward_dict:
            if reward_key != 'reward':  # Skip the consolidated reward as it's already logged above
                # Get rewards for this process and compute mean
                reward_values = all_reward_dict[reward_key][process_slice]
                if isinstance(reward_values, list):
                    reward_tensor = torch.tensor(reward_values, device=device)
                else:
                    reward_tensor = reward_values
                mean_reward = reward_tensor.view(-1, self.num_generations).mean(dim=1).mean().item()
                self._metrics[mode][f"rewards/{reward_key}"].append(mean_reward)

        # Log textual data - gather from all processes
        self._textual_logs["prompt"].extend(gather_object(prompts))
        self._textual_logs["completion"].extend(gather_object(completions))
        # Log all reward scores - both individual functions and consolidated
        for reward_key in all_reward_dict:
            # Slice rewards for this process
            reward_values = all_reward_dict[reward_key][process_slice]
            self._textual_logs["rewards"][reward_key].extend(gather_object(reward_values.tolist() if isinstance(reward_values, torch.Tensor) else reward_values))
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": None,
            "advantages": advantages,
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
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
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
        # Sync model weights to vLLM if needed
        # Skip step 0 since vLLM already has the initial weights
        if self.state.global_step > 0 and self._last_loaded_step != self.state.global_step:
            print(f"[TRAINER] Syncing weights to vLLM at step {self.state.global_step} (last loaded: {self._last_loaded_step})")
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
            print(f"[TRAINER] Weight sync complete")
            
        # Continue with normal training
        return super().training_step(model, inputs, num_items_in_batch)
    
    def _inner_training_loop(self, *args, **kwargs):
        """Override to ensure async generator is stopped when training ends"""
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            # Clean up async generator
            if self.async_generator and self._async_started:
                self.async_generator.stop()
                self._async_started = False