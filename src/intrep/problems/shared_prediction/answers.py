"""Score answers after explicit questions, without exposing answer tokens in the prefix."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def question_prefix(source, prompt, observations=()):
    ids = source.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=True, add_generation_prompt=True, return_dict=False)
    return torch.cat([*observations, source.model.encode("text", source.ids(ids))], dim=1)


def answer_loss(source, prompt, answer, observations=(), *, generate=False):
    loss, source.last_metrics, source.last_response = answer_scores(
        source, [(prompt, answer, observations)], generate=generate,
    )[0]
    return loss


def answer_scores(source, cases, *, generate=False):
    """Score equal-length answers together, returning each question in input order."""
    groups = {}
    for index, (prompt, answer, observations) in enumerate(cases):
        prefix = question_prefix(source, prompt, observations)
        targets = source.text_ids(answer) + [source.tokenizer.eos_token_id]
        groups.setdefault((prefix.shape[1], len(targets)), []).append((index, prefix, targets))
    results = [None] * len(cases)
    for (prefix_length, target_length), rows in groups.items():
        prefix = torch.cat([row[1] for row in rows], dim=0)
        expected = torch.tensor([row[2] for row in rows], dtype=torch.long, device=source.device)
        suffix = source.model.encode("text", expected[:, :-1]) if target_length > 1 else prefix[:, :0]
        hidden = source.model(torch.cat((prefix, suffix), dim=1))[:, prefix_length - 1:]
        logits = source.model.decode("text", hidden)
        losses = F.cross_entropy(logits.flatten(0, 1), expected.flatten(), reduction="none").view_as(expected).mean(1)
        correct = logits.detach().argmax(-1) == expected
        for row_index, (index, context, targets) in enumerate(rows):
            prompt, answer, _ = cases[index]
            metrics = {"answer_token_accuracy": correct[row_index].float().mean(),
                       "teacher_forced_exact": correct[row_index].all().float()}
            response = {"prompt": prompt, "expected": answer, "target_tokens": target_length,
                        "prefix_tokens": prefix_length}
            if generate:
                with torch.no_grad():
                    predicted = []
                    for _ in range(max(8, len(targets) + 4)):
                        hidden = source.model(context)[:, -1:]
                        token = int(source.model.decode("text", hidden)[0, 0].argmax())
                        if token == source.tokenizer.eos_token_id:
                            break
                        predicted.append(token)
                        context = torch.cat((context, source.model.encode("text", source.ids([token]))), dim=1)
                    text = source.tokenizer.decode(predicted, skip_special_tokens=True)
                response["answer"] = text
                metrics["exact_match"] = float(text.strip() == answer.strip())
            results[index] = losses[row_index], metrics, response
    return results
