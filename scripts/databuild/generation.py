"""Loading a generative model and running greedy generation in batches."""
import gc
import logging
import warnings

import torch

from .common import BatchFn


def model_kwargs(quant: str) -> dict:
    """Loading options: float32 on CPU; bfloat16 or a quantized load on GPU."""
    if not torch.cuda.is_available():
        if quant != "none":
            print("No GPU: ignoring --quant, loading in float32 (slow)")
        return {"torch_dtype": torch.float32}
    kwargs: dict = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if quant != "none":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quant == "8bit", load_in_4bit=quant == "4bit",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        # 8-bit layers compute in float16 and warn about it on every call.
        warnings.filterwarnings("ignore", message="MatMul8bitLt")
    return kwargs


@torch.inference_mode()
def chat_generate(model, processor, conversations: list, max_new_tokens: int) -> list[str]:
    """Greedy generation for a batch of chat conversations; returns only the new text."""
    inputs = processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in processor.batch_decode(new_tokens, skip_special_tokens=True)]


def split_on_oom(fn: BatchFn) -> BatchFn:
    """Wrap a batch function: on GPU out-of-memory, run the two halves of the batch instead."""
    def run(texts: list[str]) -> list[str]:
        try:
            return fn(texts)
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
        # The retry must run after the except block: inside it, the traceback of the error keeps
        # the tensors of the failed batch alive, and empty_cache() cannot free them.
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Out of GPU memory at batch size {len(texts)}: splitting. Lower --batch-size.")
        logging.error("Out of GPU memory at batch size %d: batch split", len(texts))
        mid = len(texts) // 2
        return run(texts[:mid]) + run(texts[mid:])
    return run
