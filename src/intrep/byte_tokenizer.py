from __future__ import annotations


class ByteTokenizer:
    pad_id = 256
    vocab_size = 257

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        payload = []
        for token_id in token_ids:
            if token_id == self.pad_id:
                continue
            if not 0 <= token_id < 256:
                raise ValueError(f"invalid byte token id: {token_id}")
            payload.append(token_id)
        return bytes(payload).decode("utf-8", errors="replace")
