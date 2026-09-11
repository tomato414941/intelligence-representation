from __future__ import annotations

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

    @torch.no_grad()
    def chat(self, messages: Sequence[dict[str, str]], *, memory: torch.Tensor | None = None,
             max_new_tokens: int = 256) -> str:
        if max_new_tokens < 1:
            raise ValueError('generation budget must be positive')
        prompt = self.prompt_ids(messages)
        memory = self.conversation_memory(prompt, memory)
        answer = []
        for _ in range(max_new_tokens):
            scores = self.text_logits(memory, [(prompt + answer)[-self.context_bytes:]])[0][-1]
            logits = scores.clone()
            # Only raw bytes and the end-of-answer token are valid outputs.
            logits[256:] = float('-inf')
            # EOS is outside the raw byte vocabulary.
            logits[EOS] = scores[EOS]
            token = int(logits.argmax())
            if token == EOS:
                break
            answer.append(token)
        return ByteTokenizer().decode(answer)
