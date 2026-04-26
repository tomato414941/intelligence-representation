from __future__ import annotations


class ByteTokenizer:
    pad_id = 256
    vocab_size = 257

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        payload = bytes(token_id for token_id in token_ids if 0 <= token_id < 256)
        return payload.decode("utf-8", errors="replace")
