from __future__ import annotations

from pathlib import Path

import torch

from intrep.problems.multimodal_agent.runtime import AgentSession
from intrep.representation.assemblies.language_agent import LanguageAgentModel
from intrep.representation.inputs.multimodal_observation import MultimodalObservation


class LanguageSession(AgentSession):
    """Conversation history and learned observation memory persist independently of replay."""

    def __init__(self, model: LanguageAgentModel, *, checkpoint_id: str, seed: int = 0) -> None:
        super().__init__(model, checkpoint_id=checkpoint_id, seed=seed)

    def reset(self) -> None:
        super().reset()
        self.messages: list[dict[str, str]] = []
        self.has_world_memory = False

    @torch.no_grad()
    def hear(self, observation: MultimodalObservation) -> None:
        self.memory = self.model.observe([observation], self.memory, step=self.step).detach()
        self.step += 1
        self.has_world_memory = True

    @torch.no_grad()
    def reply(self, text: str, *, observation: MultimodalObservation | None = None, max_new_tokens: int = 256) -> str:
        if not text.strip() or max_new_tokens < 1:
            raise ValueError("chat requires nonempty user text and a positive token budget")
        before = self.memory, self.step, self.has_world_memory
        try:
            if observation is not None:
                self.hear(observation)
            answer_messages = [*self.messages, {"role": "user", "content": text}]
            answer = self.model.chat(answer_messages, memory=self.memory if self.has_world_memory else None,
                                     max_new_tokens=max_new_tokens)
        except Exception:
            self.memory, self.step, self.has_world_memory = before
            raise
        self.messages.append({"role": "user", "content": text})
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    @torch.no_grad()
    def act(self, observation: MultimodalObservation, *, epsilon: float = 0.0, max_text_bytes: int = 16):
        decision = super().act(observation, epsilon=epsilon, max_text_bytes=max_text_bytes)
        self.has_world_memory = True
        return decision

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save({"schema_version": "intrep.language_session.v1", "checkpoint_id": self.checkpoint_id,
                    "step": self.step, "memory": self.memory.cpu(), "rng": self.generator.get_state(),
                    "messages": self.messages, "has_world_memory": self.has_world_memory}, temporary)
        temporary.replace(path)

    def restore(self, path: Path) -> None:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('schema_version') != 'intrep.language_session.v1' or payload['checkpoint_id'] != self.checkpoint_id:
            raise ValueError('session belongs to a different model checkpoint')
        memory = payload['memory'].to(self.memory.device)
        self.model._memory(memory)
        if len(memory) != 1 or payload['step'] < 0 or not torch.isfinite(memory).all():
            raise ValueError('invalid inference session state')
        messages = payload['messages']
        for index, message in enumerate(messages):
            if message['role'] != ('user' if index % 2 == 0 else 'assistant') or not isinstance(message['content'], str):
                raise ValueError('invalid conversation history')
        if len(messages) % 2:
            raise ValueError('saved conversation must end at a turn boundary')
        self.memory, self.step = memory, payload['step']
        self.generator.set_state(payload['rng'])
        self.messages, self.has_world_memory = messages, bool(payload['has_world_memory'])
