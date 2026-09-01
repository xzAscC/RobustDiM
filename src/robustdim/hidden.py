from typing import Any

import torch
from torch import Tensor


def last_token(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    idx = attention_mask.sum(dim=1) - 1
    batch = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch, idx]


def chat_text(tokenizer: Any, user: str, add_generation_prompt: bool = True) -> str:
    messages = [{"role": "user", "content": user}]
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    return str(text)
