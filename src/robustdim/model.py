from typing import Any

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from robustdim.hidden import chat_text, last_token


def apply_steer(hidden: Tensor, direction: Tensor, alpha: float) -> Tensor:
    d = direction.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
    return hidden + alpha * d


class HookedLM:
    def __init__(self, model_id: str, dtype: str = "bfloat16") -> None:
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch_dtype,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.layers = self.model.model.layers

    def _encode(self, texts: list[str]) -> dict[str, Tensor]:
        device = next(self.model.parameters()).device
        tok = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(device) for k, v in tok.items()}

    @torch.inference_mode()
    def extract(self, texts: list[str], layer: int, batch_size: int = 8) -> Tensor:
        chunks: list[Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = [
                chat_text(self.tokenizer, t) for t in texts[start : start + batch_size]
            ]
            tok = self._encode(batch)
            captured: dict[str, Tensor] = {}

            def hook(_module: Any, _inp: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                captured["h"] = tensor.detach()

            handle = self.layers[layer].register_forward_hook(hook)
            try:
                self.model(**tok)
            finally:
                handle.remove()
            chunks.append(
                last_token(captured["h"], tok["attention_mask"]).float().cpu()
            )
        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def generate(
        self,
        user: str,
        layer: int,
        direction: Tensor | None = None,
        alpha: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:
        tok = self._encode([chat_text(self.tokenizer, user)])
        handle = None
        if direction is not None and alpha != 0.0:

            def hook(_module: Any, _inp: Any, output: Any) -> Any:
                tensor = output[0] if isinstance(output, tuple) else output
                steered = apply_steer(tensor, direction, alpha)
                if isinstance(output, tuple):
                    return (steered, *output[1:])
                return steered

            handle = self.layers[layer].register_forward_hook(hook)
        try:
            generate = getattr(self.model, "generate")
            out = generate(
                **tok,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            if handle is not None:
                handle.remove()
        return str(
            self.tokenizer.decode(
                out[0, tok["input_ids"].shape[1] :], skip_special_tokens=True
            )
        )
