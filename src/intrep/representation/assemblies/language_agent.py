from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from intrep.representation.assemblies.multimodal_agent import (
    MultimodalAgentBase,
    MultimodalAgentConfig,
    MultimodalAgentModel,
)
from intrep.representation.inputs.multimodal_observation import MultimodalObservation
from intrep.sources.language.conversations import ConversationExample

WORLD_PROMPT = [{"role": "user", "content": "Which target color is indicated by the observations? Answer with one color word."}]


class LanguageAgentModel(MultimodalAgentBase):
    """A pretrained causal decoder shared by conversation, native perception and action.

    LoRA changes the decoder in every mode. Previously learned native adapters
    retain perception and forecasting while a learned residual couples the cores.
    """

    def __init__(self, backbone: nn.Module, tokenizer, native_base: dict) -> None:
        native = MultimodalAgentModel(MultimodalAgentConfig(**native_base['config']))
        native.load_state_dict(native_base['model'], strict=True)
        native.requires_grad_(False)
        super().__init__(native.config, shared_core=backbone)
        self.tokenizer = tokenizer
        self.native_base = native_base
        for name in ('observation_input', 'initial_memory', 'memory_queries', 'memory_gate',
                     'memory_norm', 'requests', 'action_output', 'image_output', 'audio_output', 'feedback_output'):
            setattr(self, name, getattr(native, name))
        self.native_core = native.core
        self.native_to_language = nn.Linear(native.config.embedding_dim, backbone.config.hidden_size, bias=False)
        self.language_to_native = nn.Linear(backbone.config.hidden_size, native.config.embedding_dim, bias=False)
        self.memory_scale = nn.Parameter(torch.tensor(0.02))
        self.native_gain = nn.Parameter(torch.tensor(0.001))
        # Frozen native knowledge initializes perception and forecasting. A small,
        # learned residual lets the same language decoder modify every native path.
        device = backbone.get_input_embeddings().weight.device
        for name, module in self.named_children():
            if name != 'core':
                module.to(device)
        for parameter in (self.initial_memory, self.memory_queries, self.memory_scale, self.native_gain):
            parameter.data = parameter.data.to(device)

    @classmethod
    def from_pretrained(cls, directory: Path, native_base: dict, *, device: str = 'cpu',
                        rank: int = 8) -> LanguageAgentModel:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        backbone = AutoModelForCausalLM.from_pretrained(
            directory, local_files_only=True, dtype=torch.bfloat16, device_map={'': device},
        )
        if getattr(backbone, 'is_loaded_in_4bit', False):
            backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
        backbone = get_peft_model(backbone, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type='CAUSAL_LM',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        ))
        return cls(backbone, tokenizer, native_base)

    def _language_run(self, sequences: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not sequences:
            raise ValueError('batch must not be empty')
        embedded = pad_sequence(list(sequences), batch_first=True)
        lengths = torch.tensor([len(row) for row in sequences], device=embedded.device)
        mask = torch.arange(embedded.shape[1], device=embedded.device)[None] < lengths[:, None]
        decoder = self.core.get_base_model().model
        hidden = decoder(inputs_embeds=embedded.to(self.core.get_input_embeddings().weight.dtype),
                         attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state.float()
        return [hidden[index, :len(row)] for index, row in enumerate(sequences)]

    def _language_memory(self, memory: torch.Tensor) -> torch.Tensor:
        return self.native_to_language(memory) * self.memory_scale

    def _run(self, sequences: Sequence[torch.Tensor], *, causal: bool = False,
             texts: Sequence[str] | None = None) -> list[torch.Tensor]:
        if not sequences:
            raise ValueError('batch must not be empty')
        embedded = pad_sequence(list(sequences), batch_first=True)
        lengths = torch.tensor([len(row) for row in sequences], device=embedded.device)
        valid = torch.arange(embedded.shape[1], device=embedded.device)[None] < lengths[:, None]
        bias = embedded.new_zeros((len(sequences), embedded.shape[1], embedded.shape[1]))
        bias.masked_fill_(~valid[:, None, :], float('-inf'))
        native = self.native_core(embedded, causal=causal, attention_logit_bias=bias)
        local = [native[index, :len(row)].float() for index, row in enumerate(sequences)]
        language_inputs = []
        for index, row in enumerate(local):
            mapped = self._language_memory(row)
            if texts is not None and texts[index]:
                mapped = torch.cat((self._embed(self._ids(texts[index])), mapped))
            language_inputs.append(mapped)
        language = self._language_run(language_inputs)
        return [row + self.native_gain.tanh() * self.language_to_native(hidden[-len(row):])
                for row, hidden in zip(local, language)]

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _embed(self, ids: Sequence[int]) -> torch.Tensor:
        return self.core.get_input_embeddings()(torch.tensor(ids, device=self.initial_memory.device)).float()

    def observe(self, observations: Sequence[MultimodalObservation], memory: torch.Tensor | None = None,
                *, step: int = 0) -> torch.Tensor:
        if memory is None:
            memory = self.new_memory(len(observations))
        self._memory(memory)
        if len(observations) != len(memory):
            raise ValueError('observation and memory batch sizes differ')
        encoded = self.observation_input.encode_many(observations, step)
        sequences = [torch.cat((row, observation, self.memory_queries)) for row, observation in zip(memory, encoded)]
        hidden = self._run(sequences, texts=[row.text for row in observations])
        proposal = torch.stack([row[-self.config.memory_tokens:] for row in hidden])
        gate = self.memory_gate(torch.cat((memory, proposal), -1)).sigmoid()
        return self.memory_norm(gate * proposal + (1 - gate) * memory)

    def prompt_ids(self, messages: Sequence[dict[str, str]]) -> list[int]:
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("a response requires a conversation ending in a user message")
        encoded = self.tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=False, return_dict=True)
        return encoded['input_ids']

    def answer_loss(self, messages: Sequence[dict[str, str]], answer: str,
                    memory: torch.Tensor | None = None) -> torch.Tensor:
        prompt = self.prompt_ids(messages)
        target = [*self._ids(answer), self.tokenizer.eos_token_id]
        embeddings = self._embed([*prompt, *target[:-1]])
        offset = len(prompt) - 1
        if memory is not None:
            self._memory(memory)
            if len(memory) != 1:
                raise ValueError("answer loss accepts one memory row")
            embeddings = torch.cat((self._language_memory(memory[0]), embeddings))
            offset += self.config.memory_tokens
        hidden = self._language_run([embeddings])[0][offset:]
        head = self.core.get_output_embeddings()
        logits = head(hidden.to(head.weight.dtype)).float()
        return F.cross_entropy(logits, torch.tensor(target, device=logits.device))

    def conversation_loss(self, examples: Sequence[ConversationExample]) -> torch.Tensor:
        if not examples:
            raise ValueError("conversation batch must not be empty")
        return torch.stack([self.answer_loss(row.prompt(), row.answer) for row in examples]).mean()

    def text_loss(self, memory: torch.Tensor, targets: Sequence[str]) -> torch.Tensor:
        if len(targets) != len(memory):
            raise ValueError("text and memory batch sizes differ")
        return torch.stack([self.answer_loss(WORLD_PROMPT, target, row[None])
                            for row, target in zip(memory, targets)]).mean()

    @torch.no_grad()
    def chat(self, messages: Sequence[dict[str, str]], *, memory: torch.Tensor | None = None,
             max_new_tokens: int = 256) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        ids = torch.tensor([self.prompt_ids(messages)], device=self.initial_memory.device)
        kwargs = {}
        if memory is not None:
            self._memory(memory)
            if len(memory) != 1:
                raise ValueError("chat accepts one memory row")
            kwargs["inputs_embeds"] = torch.cat((self._language_memory(memory),
                                                  self.core.get_input_embeddings()(ids)), dim=1).to(
                                                      self.core.get_input_embeddings().weight.dtype)
            kwargs["attention_mask"] = torch.ones(kwargs["inputs_embeds"].shape[:2], device=ids.device, dtype=torch.long)
        else:
            kwargs["attention_mask"] = torch.ones_like(ids)
        output = self.core.generate(input_ids=ids, **kwargs, max_new_tokens=max_new_tokens,
                                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        return self.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_text(self, memory: torch.Tensor, *, max_bytes: int = 128) -> list[str]:
        return [self.chat(WORLD_PROMPT, memory=row[None], max_new_tokens=max_bytes)
                .encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore') for row in memory]

    def learned_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters() if value.requires_grad}

    def restore_learned_state(self, state: dict[str, torch.Tensor]) -> None:
        parameters = {name: value for name, value in self.named_parameters() if value.requires_grad}
        if state.keys() != parameters.keys() or any(state[name].shape != value.shape for name, value in parameters.items()):
            raise ValueError("learned state does not match this backbone and adapter configuration")
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(state[name])
