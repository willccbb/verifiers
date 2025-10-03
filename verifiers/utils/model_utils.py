from importlib.util import find_spec
from typing import Any, Callable, List, Dict, Union

from verifiers.utils.image_utils import _base64_to_pil

import torch  # type: ignore[unresolved-import]
import torch.nn as nn  # type: ignore[unresolved-import]
from transformers import (  # type: ignore[unresolved-import]
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.tokenization_utils_base import (  # type: ignore[unresolved-import]
    PreTrainedTokenizerBase, ProcessorMixin
)

class _ForwardRedirection:
    """Implements the `forward-redirection`.

    Taken from Pytorch-lightning: https://github.com/Lightning-AI/pytorch-lightning/blob/02311d03fb982560246eead7c08104481fac9579/src/lightning/pytorch/strategies/strategy.py#L602

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(
        self,
        wrapper_module: nn.Module,
        original_module: nn.Module,
        method: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.

        """
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(
        self, wrapper_module: nn.Module, original_module: nn.Module
    ) -> None:
        pass

    def on_after_outer_forward(
        self, wrapper_module: nn.Module, original_module: nn.Module
    ) -> None:
        pass


def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None


def get_model(
    model_name: str,
    use_liger: bool = True,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    if is_liger_available() and use_liger:
        print("Using Liger kernel")
        from liger_kernel.transformers import (  # type: ignore[unresolved-import]
            AutoLigerKernelForCausalLM,
        )

        return AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def get_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not hasattr(tokenizer, "chat_template"):
        raise ValueError(
            f"Tokenizer for model {model_name} does not have chat_template attribute, \
                            and could not find a tokenizer with the same name as the model with suffix \
                            '-Instruct'. Please provide a tokenizer with the chat_template attribute."
        )
    return tokenizer


def get_model_and_tokenizer(
    model_name: str, use_liger: bool = True, model_kwargs: dict[str, Any] | None = None
) -> tuple[Any, Any]:
    model = get_model(model_name, use_liger, model_kwargs)
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer

def encode_chat_with_processor(
    conversation: List[Dict],
    processing_class: Union[PreTrainedTokenizerBase, ProcessorMixin],
    add_generation_prompt: bool = False,
    add_special_tokens: bool = False,
) -> List[int]:
    """
    Apply chat template and return token IDs, handling both tokenizer and processor.
    Supports base64-encoded images in the conversation.
    """

    if isinstance(processing_class, ProcessorMixin):
        prompt_text = processing_class.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

        images = []
        for msg in conversation:
            for c in msg.get("content", []):
                if c.get("type") == "image_url":
                    pil_img = _base64_to_pil(c["image_url"]["url"])
                    images.append(pil_img)

        inputs = processing_class(
            text=[prompt_text],
            images=images if images else None,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        return inputs["input_ids"][0].tolist(), inputs["image_grid_thw"][0].tolist(), inputs["pixel_values"].tolist()

    else:
        prompt_ids : List[int] = processing_class.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
        )
        return prompt_ids,None,None
    
def encode_text_with_processor(
    text: str,
    processing_class: Union[PreTrainedTokenizerBase, ProcessorMixin],
) -> tuple[list[int], Any, Any]:
    """
    Encode plain text and return token IDs, handling both tokenizer and processor.
    """
    if isinstance(processing_class, ProcessorMixin):
        inputs = processing_class(
            text=[text],
            images=None,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0].tolist()
        image_grid = inputs.get("image_grid_thw", [None])[0].tolist()
        pixel_values = inputs.get("pixel_values", [None]).tolist()
        return input_ids, image_grid, pixel_values
    else:
        prompt_ids: list[int] = processing_class.encode(
            text
        )
        return prompt_ids, None, None