from __future__ import annotations

import torch
from torch import nn

from intrep.representation.assemblies.shared_predictor import SharedPredictor


class LfmCore(nn.Module):
    """LFM's causal convolution/attention body, without vocabulary or output heads."""

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        if body.embed_tokens is not None:
            raise ValueError("detach the text embedding from the core before assembly")
        self.body = body

    def forward(self, embeddings: torch.Tensor, *, attention_mask=None, position_ids=None) -> torch.Tensor:
        # Training and head changes never reuse stale attention/convolution caches.
        return self.body(inputs_embeds=embeddings, attention_mask=attention_mask,
                         position_ids=position_ids, use_cache=False).last_hidden_state


def split_lfm(model: nn.Module) -> SharedPredictor:
    """Move the pretrained text heads out of the body, preserving their weight tie."""
    if model.config.model_type != "lfm2":
        raise ValueError("this assembly requires an LFM2-family checkpoint")
    text_input = model.get_input_embeddings()
    text_output = model.get_output_embeddings()
    body = model.model
    body.embed_tokens = None
    predictor = SharedPredictor(LfmCore(body), model.config.hidden_size)
    predictor.attach_input("text", text_input)
    predictor.attach_output("text", text_output)
    return predictor


def load_lfm(path: str, *, device: str = "cpu", dtype: torch.dtype = torch.float32) -> SharedPredictor:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=dtype, attn_implementation="eager",
    ).to(device)
    return split_lfm(model)


def create_lfm(config: dict, *, device: str = "cpu", attention_implementation: str = "eager") -> SharedPredictor:
    from transformers import Lfm2Config, Lfm2ForCausalLM

    configuration = Lfm2Config.from_dict(config, attn_implementation=attention_implementation)
    return split_lfm(Lfm2ForCausalLM(configuration).to(device))
