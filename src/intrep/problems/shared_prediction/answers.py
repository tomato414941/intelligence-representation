"""Score answers after explicit questions, without exposing answer tokens in the prefix."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def question_prefix(source, prompt, observations=()):
    ids = source.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=True, add_generation_prompt=True, return_dict=False)
    return torch.cat([*observations, source.model.encode("text", source.ids(ids))], dim=1)


def answer_loss(source, prompt, answer, observations=(), *, generate=False):
    prefix = question_prefix(source, prompt, observations)
    targets = source.text_ids(answer) + [source.tokenizer.eos_token_id]
    suffix = source.model.encode("text", source.ids(targets[:-1])) if len(targets) > 1 else prefix[:, :0]
    hidden = source.model(torch.cat((prefix, suffix), dim=1))[:, prefix.shape[1] - 1:]
    logits = source.model.decode("text", hidden)[0]
    expected = source.ids(targets)[0]
    loss = F.cross_entropy(logits, expected)
    chosen = logits.detach().argmax(-1)
    source.last_metrics = {"answer_token_accuracy": (chosen == expected).float().mean(),
                           "teacher_forced_exact": (chosen == expected).all().float()}
    source.last_response = {"prompt": prompt, "expected": answer, "target_tokens": len(targets),
                            "prefix_tokens": prefix.shape[1]}
    if generate:
        with torch.no_grad():
            context, predicted = prefix, []
            for _ in range(max(8, len(targets) + 4)):
                hidden = source.model(context)[:, -1:]
                token = int(source.model.decode("text", hidden)[0, 0].argmax())
                if token == source.tokenizer.eos_token_id:
                    break
                predicted.append(token)
                context = torch.cat((context, source.model.encode("text", source.ids([token]))), dim=1)
            text = source.tokenizer.decode(predicted, skip_special_tokens=True)
        source.last_response["answer"] = text
        source.last_metrics["exact_match"] = float(text.strip() == answer.strip())
    return loss
