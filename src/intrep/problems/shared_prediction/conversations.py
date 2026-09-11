"""Assistant supervision with conversation boundaries and overlapping context."""
from __future__ import annotations

import json

import torch
from torch.nn import functional as F

from intrep.problems.shared_prediction.sources import TextSource


class AssistantConversationSource(TextSource):
    def __init__(self, *args):
        super().__init__(*args)
        self.window = self.config.get("conversation_tokens", 2048)
        self.overlap = self.config.get("conversation_overlap", self.window // 2)
        if not isinstance(self.window, int) or not 2 <= self.window:
            raise ValueError("conversation_tokens must be an integer of at least two")
        if not isinstance(self.overlap, int) or not 1 <= self.overlap < self.window:
            raise ValueError("conversation_overlap must retain context within the window")
        if "{% generation" not in (self.tokenizer.chat_template or "") and "{%- generation" not in (self.tokenizer.chat_template or ""):
            raise ValueError("assistant supervision requires generation spans in the chat template")
        self.pending_mask = []
        self.position = 0
        self.record_id = self.group_id = None
        self.input_tokens = self.supervised_tokens = self.windows = self.skipped = 0

    def next_record(self):
        epoch = self.stream.epochs
        while True:
            if self.position >= len(self.pending):
                if self.stream.epochs > epoch + 1:
                    raise ValueError("conversation population has no assistant targets")
                row = json.loads(self.stream.next())
                encoded = self.tokenizer.apply_chat_template(
                    row["messages"], tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_assistant_tokens_mask=True,
                )
                self.pending = list(encoded["input_ids"])
                self.pending_mask = list(encoded["assistant_masks"])
                if len(self.pending) != len(self.pending_mask):
                    raise ValueError("assistant mask must align with the rendered conversation")
                self.tokens += len(self.pending)
                self.record_id = row.get("id", f"byte:{self.stream.offset}")
                self.group_id = row.get("group_id", self.record_id)
                self.position = 0
                if not any(self.pending_mask[1:]):
                    self.position = len(self.pending)
                    self.skipped += 1
                    continue
            start = max(0, self.position - self.overlap)
            end = min(len(self.pending), start + self.window)
            # Overlap supplies context, but every assistant target is scored once.
            mask = [bool(value) and index >= max(1, self.position)
                    for index, value in enumerate(self.pending_mask[start:end], start)]
            self.position = end
            mask[0] = False
            if not any(mask):
                continue
            self.windows += 1
            self.input_tokens += end - start
            return {"tokens": self.ids(self.pending[start:end]),
                    "mask": torch.tensor([mask], dtype=torch.bool, device=self.device),
                    "index": self.record_id, "group": self.group_id, "start": start, "end": end}

    def record_loss(self, record):
        tokens, mask = record["tokens"], record["mask"][:, 1:]
        hidden = self.model(self.model.encode("text", tokens[:, :-1]))
        logits = self.model.decode("text", hidden[mask].unsqueeze(0))[0]
        targets = tokens[:, 1:][mask]
        loss = F.cross_entropy(logits, targets)
        self.supervised_tokens += targets.numel()
        self.last_group = record["group"]
        self.last_metrics = {"answer_token_accuracy": (logits.detach().argmax(-1) == targets).float().mean()}
        self.last_response = {"target_tokens": targets.numel(), "input_tokens": tokens.numel(),
                              "group": record["group"], "window": [record["start"], record["end"]]}
        return loss

    def state_dict(self):
        return {**super().state_dict(), "pending_mask": list(self.pending_mask), "position": self.position,
                "record_id": self.record_id, "group_id": self.group_id,
                "input_tokens": self.input_tokens, "supervised_tokens": self.supervised_tokens,
                "windows": self.windows, "skipped": self.skipped}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.pending_mask = list(state["pending_mask"])
        for name in ("position", "record_id", "group_id", "input_tokens", "supervised_tokens", "windows", "skipped"):
            setattr(self, name, state[name])
        if len(self.pending_mask) != len(self.pending) or not 0 <= self.position <= len(self.pending):
            raise ValueError("invalid conversation cursor or assistant mask")

    def provenance(self):
        return {**super().provenance(), "objective": "assistant spans including end-of-turn tokens",
                "context_tokens": self.window, "overlap_tokens": self.overlap,
                "long_conversations": "overlapping windows; no assistant targets discarded or repeated",
                "unanswered_records": "read and counted, but cannot supply an assistant loss"}

    def progress(self):
        return {"stream": self.stream.state_dict(), "read_tokens": self.tokens,
                "input_tokens_including_overlap": self.input_tokens, "supervised_tokens": self.supervised_tokens,
                "windows": self.windows, "records_without_targets": self.skipped}
