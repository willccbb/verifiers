# adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py

from dataclasses import dataclass, field
from typing import List, Optional, Union

import transformers  # type: ignore[unresolved-import]
from packaging import version
from transformers import TrainingArguments  # type: ignore[unresolved-import]


@dataclass
class GRPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`GRPOTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    """

    if version.parse(transformers.__version__) <= version.parse("4.50.3"):
        from transformers.training_args import (  # type: ignore[unresolved-import]
            _VALID_DICT_FIELDS,
        )

        _VALID_DICT_FIELDS.append("model_init_kwargs")
    else:
        _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + [
            "model_init_kwargs"
        ]

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    # Common TrainingArguments surfaced here for better typing in our tooling
    output_dir: str = field(
        default="",
        metadata={"help": "Where to store artifacts and checkpoints."},
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "An optional experiment name for logging."},
    )
    lr_scheduler_type: Optional[str] = field(
        default=None,
        metadata={"help": "Learning rate scheduler type."},
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup over warmup_steps."},
    )
    max_steps: int = field(
        default=-1,
        metadata={
            "help": "Total number of training steps to perform. -1 for full epochs."
        },
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bfloat16 precision."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for clipping."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device for training."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to accumulate before backward/update."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing to save memory."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "When to save checkpoints (no, steps, epoch)."},
    )
    save_steps: int = field(
        default=500,
        metadata={
            "help": "Save checkpoint every X updates steps when save_strategy=steps."
        },
    )
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": "If True, save only model weights (not optimizer/scheduler)."
        },
    )
    logging_steps: int = field(
        default=500,
        metadata={"help": "Log every X updates steps."},
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={"help": "Whether to log on each node in multi-node setup."},
    )
    report_to: Optional[Union[str, List[str]]] = field(
        default=None,
        metadata={"help": "Integration to report results and logs to (e.g., 'wandb')."},
    )

    # Parameters that control the model and reference model
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: int = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    rollout_filter_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Ratio of rollouts to keep after filtering. Must be in the range (0.0, 1.0]. For example, if "
            "set to 0.5, only the top 50% of rollouts based on their rewards will be kept for training. If set to "
            "1.0 (default), all rollouts are kept."
        },
    )
    # Deprecated, use max_seq_len instead
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated. Use `max_seq_len` instead."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Parameters that control generation
    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * gradient_accumulation_steps`."
        },
    )
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of optimization steps per generation. If `None`, it defaults to gradient_accumulation_steps."
        },
    )
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens to generate per turn."},
    )
    max_seq_len: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum number of tokens to generate per turn."},
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty (default 0.0)"},
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Frequency penalty (default 0.0)"},
    )
    max_data_workers: int = field(
        default=8,
        metadata={
            "help": "Maximum number of processes to use for filtering the dataset."
        },
    )
    max_concurrent: int = field(
        default=1024,
        metadata={"help": "Maximum number of concurrent requests to the environment."},
    )
    # Async generation parameters
    num_batches_ahead: int = field(
        default=1,
        metadata={
            "help": "Number of batches to generate ahead. Higher values can improve GPU utilization but use more memory. "
            "Set to 0 for synchronous generation (submit and wait immediately, no look-ahead)."
        },
    )
    async_generation_timeout: float = field(
        default=600.0,
        metadata={
            "help": "Timeout in seconds for async generation. If a batch doesn't complete within this time, "
            "a TimeoutError is raised."
        },
    )
    async_max_queue_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of batches that can be queued for async generation. If None, defaults to "
            "2 * num_batches_ahead."
        },
    )

    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={
            "help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."
        },
    )

    # Parameters that control the vLLM server
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to."},
    )
    vllm_server_timeout: float = field(
        default=300.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )
    # Parameters that control the training
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.001,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={
            "help": "Number of iterations per batch (denoted as μ in the algorithm)."
        },
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    delta: Optional[float] = field(
        default=None,
        metadata={
            "help": "If set to a float value (e.g., 2.0), enables the upper clipping bound in two-sided GRPO loss. If None (default), the standard GRPO clipping is used. Recommended to be > 1 + epsilon when enabled."
        },
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    scale_rewards: bool = field(
        default=False,
        metadata={
            "help": "Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), "
            "the rewards are normalized by the standard deviation, ensuring they have unit variance. If `False`, no "
            "scaling is applied. The Dr. GRPO paper recommends not scaling the rewards, as scaling by the standard "
            "deviation introduces a question-level difficulty bias."
        },
    )
    loss_type: str = field(
        default="dr_grpo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are `grpo`, `bnpo`, and `dr_grpo`. "
            "`'grpo'`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to "
            "length bias—this approach tends to prefer shorter completions with positive advantages and longer ones "
            "with negative advantages. "
            "`'bnpo'`: Aggregates token-level losses by normalizing number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss. "
            "`'dr_grpo'`: Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`."
        },
    )
    mask_env_responses: bool = field(
        default=True,
        metadata={
            "help": "Whether to mask the environment responses. If `True`, the environment responses are masked, "
            "preventing them from being incorrectly penalized and introducing noise during training."
        },
    )
    mask_truncated_completions: bool = field(
        default=True,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )
    zero_truncated_completions: bool = field(
        default=False,
        metadata={"help": "Whether to give zero reward to truncated completions."},
    )
    sync_ref_model: bool = field(
        default=True,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.5,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=100,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of completions to print with `rich`. If `None`, all completions are logged."
        },
    )
    wandb_log_unique_prompts: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, "
            "all prompts are logged."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        num_processes = self.world_size
        # The current default effective batch size
        if (
            self.generation_batch_size is not None
            and self.steps_per_generation is not None
        ):
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps

        if self.generation_batch_size is None:
            self.generation_batch_size = (
                self.per_device_train_batch_size
                * num_processes
                * self.steps_per_generation
            )

        # Type narrowing for static checkers
        assert self.generation_batch_size is not None

        if (
            self.generation_batch_size
            % self.per_device_train_batch_size
            * num_processes
            != 0
        ):
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                f"({self.per_device_train_batch_size * num_processes})."
            )

        self.steps_per_generation = self.generation_batch_size // (
            self.per_device_train_batch_size * num_processes
        )

        assert self.generation_batch_size is not None
        # Check if the effective batch size can be divided by the number of generations
        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )
        possible_values = [
            n_gen
            for n_gen in range(2, self.generation_batch_size + 1)
            if (self.generation_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {self.per_device_train_batch_size} x "
                f"{self.steps_per_generation}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )

        if not (0 < self.rollout_filter_ratio <= 1.0):
            raise ValueError(
                f"rollout_filter_ratio must be in (0, 1], got {self.rollout_filter_ratio}."
            )

        groups = (
            self.generation_batch_size // self.num_generations
        )  # prompts per step (since each prompt has G generations)

        top_n = int(self.rollout_filter_ratio * groups)

        if top_n <= 0:
            # Smallest workable ratio is 1/groups
            min_ratio = 1.0 / max(groups, 1)
            raise ValueError(
                "rollout_filter_ratio is too small: after filtering, no groups would remain. "
                f"Given B={self.generation_batch_size}, G={self.num_generations} -> groups={groups}, set rollout_filter_ratio >= {min_ratio:.6f} "
                "(or increase generation_batch_size)."
            )

        K = top_n * self.num_generations  # kept total samples after filtering

        # Sharding evenly across processes
        if K % self.world_size != 0:
            raise ValueError(
                "After rollout filtering, kept samples must be divisible by world_size. "
                f"Got K={K}, world_size={self.world_size}. "
            )

        # Per-process kept
        K_per_proc = K // self.world_size

        # Spliting per-process batch into gradient_accumulation_steps equal chunks later.
        if K_per_proc % self.gradient_accumulation_steps != 0:
            raise ValueError(
                "Per-process kept batch size must be divisible by gradient_accumulation_steps. "
                f"Got K_per_proc={K_per_proc}, gradient_accumulation_steps={self.gradient_accumulation_steps}. "
            )

        if self.eval_strategy != "no":
            global_eval_batch_size = self.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen
                for n_gen in range(2, global_eval_batch_size + 1)
                if (global_eval_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {self.per_device_eval_batch_size}) must be "
                    f"evenly divisible by the number of generations per prompt ({self.num_generations}). Given the "
                    "current global eval batch size, the valid values for the number of generations are: "
                    f"{possible_values}."
                )
