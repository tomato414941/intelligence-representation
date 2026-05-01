from __future__ import annotations

import random
from pathlib import Path


def read_split_text_corpora(
    corpus_paths: list[Path],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[str, str]:
    if not corpus_paths:
        raise ValueError("at least one corpus path is required")
    train_texts = []
    eval_texts = []
    for path in corpus_paths:
        text = path.read_text(encoding="utf-8")
        if not text:
            raise ValueError(f"corpus must not be empty: {path}")
        train_text, eval_text = split_text_corpus(text, eval_ratio=eval_ratio)
        train_texts.append(train_text.strip())
        eval_texts.append(eval_text.strip())
    randomizer = random.Random(seed)
    randomizer.shuffle(train_texts)
    randomizer.shuffle(eval_texts)
    return join_text_documents(train_texts), join_text_documents(eval_texts)


def read_text_corpora(corpus_paths: list[Path], *, seed: int) -> str:
    if not corpus_paths:
        raise ValueError("at least one corpus path is required")
    texts = []
    for path in corpus_paths:
        text = path.read_text(encoding="utf-8")
        if not text:
            raise ValueError(f"corpus must not be empty: {path}")
        texts.append(text.strip())
    random.Random(seed).shuffle(texts)
    return join_text_documents(texts)


def join_text_documents(texts: list[str]) -> str:
    if not texts:
        raise ValueError("at least one text document is required")
    return "\n<|endoftext|>\n".join(texts) + "\n"


def split_text_corpus(text: str, *, eval_ratio: float) -> tuple[str, str]:
    if not text:
        raise ValueError("corpus must not be empty")
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    split_index = int(len(text) * (1.0 - eval_ratio))
    train_text = text[:split_index]
    eval_text = text[split_index:]
    if not train_text or not eval_text:
        raise ValueError("eval_ratio produced an empty train or eval split")
    return train_text, eval_text
