from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from intrep.representation.assemblies.multimodal_agent import MultimodalAgentModel
from intrep.representation.inputs.multimodal_observation import (
    EOS,
    MultimodalObservation,
)
from intrep.sources.language.byte_tokenizer import ByteTokenizer
from intrep.sources.language.conversations import ConversationExample


class LanguageAgentModel(MultimodalAgentModel):
    """Conversation and native experience use one trainable transformer and byte vocabulary."""

    context_bytes = 1024

    def prompt_ids(self, messages: Sequence[dict[str, str]]) -> list[int]:
        if not messages or messages[-1]['role'] != 'user':
            raise ValueError('a conversation prompt must end with a user message')
        parts = []
        for message in messages:
            if message['role'] not in ('system', 'user', 'assistant') or not isinstance(message['content'], str):
                raise ValueError('invalid conversation message')
            parts.append(f"<{message['role']}>\n{message['content']}\n")
        return ByteTokenizer().encode(''.join(parts) + '<assistant>\n')[-self.context_bytes:]

    def conversation_memory(self, prompt: Sequence[int], memory: torch.Tensor | None = None) -> torch.Tensor:
        # Use the same observation encoder and recurrent state update as image/audio/action feedback.
        return self.observe([MultimodalObservation(text=ByteTokenizer().decode(list(prompt)))], memory)

    def answer_loss(self, messages: Sequence[dict[str, str]], answer: str,
                    memory: torch.Tensor | None = None) -> torch.Tensor:
        prompt = self.prompt_ids(messages)
        memory = self.conversation_memory(prompt, memory)
        target = ByteTokenizer().encode(answer)
        # Train every answer byte in bounded windows; EOS is supervised only at its actual end.
        losses = []
        for start in range(0, len(target) + 1, self.context_bytes):
            labels = (target + [EOS])[start:start + self.context_bytes]
            prefix = (prompt + target[:start])[-self.context_bytes:]
            logits = self.text_logits(memory, [prefix + labels[:-1]])[0]
            losses.append(F.cross_entropy(logits[len(prefix):], torch.tensor(labels, device=memory.device), reduction='sum'))
        return torch.stack(losses).sum() / (len(target) + 1)

    def conversation_loss(self, examples: Sequence[ConversationExample]) -> torch.Tensor:
        if not examples:
            raise ValueError('conversation batch must not be empty')
        return torch.stack([self.answer_loss(row.prompt(), row.answer) for row in examples]).mean()

    def pretraining_loss(self, blocks: torch.Tensor) -> torch.Tensor:
        if blocks.ndim != 2 or min(blocks.shape) < 1:
            raise ValueError('pretraining requires a nonempty batch of token blocks')
        memory = self.new_memory(len(blocks))
        logits = self.text_logits(memory, blocks[:, :-1].tolist())
        return F.cross_entropy(torch.cat(logits), blocks.to(memory.device).flatten())

    def _answer_prefix(self, prompt: list[int], answer: list[int]) -> list[int]:
        start = len(answer) // self.context_bytes * self.context_bytes
        return (prompt + answer[:start])[-self.context_bytes:] + answer[start:]

    @torch.no_grad()
    def complete(self, prefix: str, *, max_new_tokens: int = 256, temperature: float = 0.0,
                 top_p: float = 1.0, generator: torch.Generator | None = None) -> str:
        return self._generate(ByteTokenizer().encode(prefix)[-self.context_bytes:],
                              self.new_memory(), max_new_tokens=max_new_tokens, temperature=temperature,
                              top_p=top_p, generator=generator)

    @torch.no_grad()
    def chat(self, messages: Sequence[dict[str, str]], *, memory: torch.Tensor | None = None,
             max_new_tokens: int = 256) -> str:
        if max_new_tokens < 1:
            raise ValueError('generation budget must be positive')
        prompt = self.prompt_ids(messages)
        memory = self.conversation_memory(prompt, memory)
        return self._generate(prompt, memory, max_new_tokens=max_new_tokens)

    def _generate(self, prompt: list[int], memory: torch.Tensor, *, max_new_tokens: int,
                  temperature: float = 0.0, top_p: float = 1.0, generator: torch.Generator | None = None) -> str:
        if max_new_tokens < 1:
            raise ValueError('generation budget must be positive')
        if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError('invalid temperature or nucleus probability')
        answer = []
        pending, lower, upper = 0, 0x80, 0xBF
        for index in range(max_new_tokens):
            scores = self.text_logits(memory, [self._answer_prefix(prompt, answer)])[0][-1]
            logits = torch.full_like(scores, float('-inf'))
            if pending:
                logits[lower:upper + 1] = scores[lower:upper + 1]
            else:
                logits[:0x80] = scores[:0x80]
                logits[EOS] = scores[EOS]
                # A leading byte must leave enough budget to complete its codepoint.
                remaining = max_new_tokens - index
                for needed, start, stop in ((2, 0xC2, 0xE0), (3, 0xE0, 0xF0), (4, 0xF0, 0xF5)):
                    if remaining >= needed:
                        logits[start:stop] = scores[start:stop]
            if temperature == 0:
                token = int(logits.argmax())
            else:
                probabilities, indices = (logits.float() / temperature).softmax(-1).sort(descending=True)
                keep = probabilities.cumsum(-1) - probabilities < top_p
                probabilities = probabilities * keep
                choice = torch.multinomial(probabilities.cpu(), 1, generator=generator)
                token = int(indices[int(choice)])
            if token == EOS:
                break
            answer.append(token)
            if pending:
                pending -= 1
                lower, upper = 0x80, 0xBF
            elif token >= 0xC2:
                pending = 1 if token < 0xE0 else 2 if token < 0xF0 else 3
                lower = 0xA0 if token == 0xE0 else 0x90 if token == 0xF0 else 0x80
                upper = 0x9F if token == 0xED else 0x8F if token == 0xF4 else 0xBF
        return bytes(answer).decode('utf-8')
