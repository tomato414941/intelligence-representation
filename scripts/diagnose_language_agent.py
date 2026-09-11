"""Bounded overfit and replay controls for the single-core language path."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from intrep.experience.multimodal.records import selected_episodes
from intrep.problems.language_agent.training import load_checkpoint
from intrep.problems.multimodal_agent.training import (
    MultimodalTrainingConfig,
    episode_loss,
)
from intrep.representation.inputs.multimodal_observation import EOS
from intrep.sources.language.conversations import ChatMessage, ConversationExample

# Diagnostic examples are not a general language benchmark.
PAIRS = [
    ('Name the color of snow.', 'Snow is white.'),
    ('What do birds use to fly?', 'Birds fly with their wings.'),
    ('What is two plus three?', 'Two plus three is five.'),
    ('Say good night.', 'Good night. Sleep well.'),
    ('犬は何と鳴きますか。', '犬はワンと鳴きます。'),
    ('朝の挨拶をしてください。', 'おはようございます。'),
    ('「I like tea.」を日本語に訳してください。', '私はお茶が好きです。'),
    ('二つのリンゴに三つ加えると何個ですか。', 'リンゴは五個です。'),
]


def examples():
    return [ConversationExample(f'diagnostic-{i}', f'diagnostic-{i}',
                                (ChatMessage('user', prompt), ChatMessage('assistant', answer)),
                                'authored-language-diagnostic-v1') for i, (prompt, answer) in enumerate(PAIRS)]


@torch.no_grad()
def probe(model, rows):
    model.eval()
    result, correct, total = [], 0, 0
    for row in rows:
        prompt = model.prompt_ids(row.prompt())
        memory = model.conversation_memory(prompt)
        labels = [*row.answer.encode(), EOS]
        logits = model.text_logits(memory, [prompt + labels[:-1]])[0][len(prompt):]
        predicted = logits.argmax(-1).tolist()
        correct += sum(a == b for a, b in zip(predicted, labels))
        total += len(labels)
        output = model.chat(row.prompt(), max_new_tokens=len(labels) + 12)
        result.append({'prompt': row.prompt(), 'target': row.answer, 'generated': output,
                       'exact': output == row.answer,
                       'loss': float(F.cross_entropy(logits, torch.tensor(labels, device=memory.device)))})
    return {'teacher_byte_accuracy': correct / total, 'exact_generations': sum(row['exact'] for row in result),
            'cases': result}


@torch.no_grad()
def native_accuracy(model, episodes):
    model.eval()
    correct = total = 0
    for episode in episodes:
        memory = model.new_memory()
        for step, observation in enumerate(episode.observations[:-1]):
            memory = model.observe([observation], memory, step=step)
            correct += int(model.policy(memory).argmax()) == episode.teacher_actions[step]
            total += 1
    return {'correct': correct, 'total': total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = examples()
    (args.output / 'examples.json').write_text(json.dumps(PAIRS, ensure_ascii=False, indent=2) + '\n')
    train_episodes = selected_episodes(args.selection, 'train')
    validation = selected_episodes(args.selection, 'validation')[:8]
    results = {}
    for condition in ('language_only', 'with_native_replay'):
        model, _ = load_checkpoint(args.checkpoint, device=args.device)
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        language_rng = torch.Generator().manual_seed(61)
        native_rng = torch.Generator().manual_seed(62)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.95), weight_decay=0.1)
        before = probe(model, rows)
        before_native = native_accuracy(model, validation)
        started = time.perf_counter()
        history = []
        for step in range(1, args.steps + 1):
            model.train()
            batch = [rows[index] for index in torch.randperm(len(rows), generator=language_rng)[:4]]
            scale = min(step / 20, 1.0) * (0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2)
            for group in optimizer.param_groups:
                group['lr'] = 0.0003 * scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
                loss = model.conversation_loss(batch)
                if condition == 'with_native_replay':
                    episodes = [train_episodes[index] for index in torch.randperm(len(train_episodes), generator=native_rng)[:2]]
                    native, _ = episode_loss(model, episodes, MultimodalTrainingConfig(model=model.config))
                    loss = loss + native
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite diagnostic loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 100 == 0 or step == args.steps:
                measured = probe(model, rows)
                row = {'step': step, 'loss': float(loss.detach()), 'elapsed_seconds': time.perf_counter() - started,
                       **{key: value for key, value in measured.items() if key != 'cases'}}
                print(json.dumps({'condition': condition, **row}), flush=True)
                history.append(row)
                if measured['exact_generations'] == len(rows):
                    break
        after = probe(model, rows)
        results[condition] = {'before': before, 'after': after, 'before_native': before_native,
                              'after_native': native_accuracy(model, validation), 'history': history,
                              'changed_core_tensors': sum(not torch.equal(initial[name], value) for name, value in model.state_dict().items() if name.startswith('core.'))}
        # Diagnostic memorization weights are retained separately from production checkpoints.
        torch.save(model.state_dict(), args.output / f'{condition}.pt')
        (args.output / 'result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        del model, initial, optimizer
        if args.device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
