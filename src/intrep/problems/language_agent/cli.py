from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from intrep.experience.multimodal.records import read_audio, write_selection
from intrep.problems.language_agent.evaluation import evaluate
from intrep.problems.language_agent.runtime import LanguageSession
from intrep.problems.language_agent.training import (
    LanguageTrainingConfig,
    load_checkpoint,
    train,
)
from intrep.problems.multimodal_agent.runtime import (
    checkpoint_identity,
    rollout,
    save_decision,
)
from intrep.representation.inputs.multimodal_observation import MultimodalObservation
from intrep.worlds.gridworld.multimodal import MultimodalNavigationWorld


def main():
    parser = argparse.ArgumentParser(description='Language-capable multimodal predictive agent')
    parser.add_argument('--threads', type=int, default=2)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('chat', 'act', 'train', 'rollout', 'cycle', 'evaluate'):
        sub = commands.add_parser(command)
        sub.add_argument('--base', type=Path, required=True)
        sub.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        sub.add_argument('--output', type=Path, required=True)
        if command in ('chat', 'act', 'rollout', 'cycle', 'evaluate'):
            sub.add_argument('--checkpoint', type=Path, required=True)
        if command in ('chat', 'act'):
            sub.add_argument('--text', default='')
            sub.add_argument('--image', type=Path)
            sub.add_argument('--audio', type=Path)
            sub.add_argument('--previous-action', type=int)
            sub.add_argument('--feedback', type=float, nargs=3)
            sub.add_argument('--session', type=Path)
            if command == 'chat':
                sub.add_argument('--max-new-tokens', type=int, default=256)
            else:
                sub.add_argument('--max-text-bytes', type=int, default=16)
        if command in ('train', 'cycle'):
            sub.add_argument('--selection', type=Path, action='append', required=True)
            sub.add_argument('--conversations', type=Path, required=True)
            sub.add_argument('--steps', type=int, default=300)
            sub.add_argument('--batch-size', type=int, default=2)
        if command == 'evaluate':
            sub.add_argument('--selection', type=Path, required=True)
            sub.add_argument('--validation-conversations', type=Path, required=True)
        if command == 'train':
            sub.add_argument('--native-checkpoint', type=Path)
            sub.add_argument('--initialize', type=Path)
            sub.add_argument('--resume', action='store_true')
            sub.add_argument('--learning-rate', type=float, default=0.0001)
            sub.add_argument('--rank', type=int, default=8)
        if command in ('rollout', 'cycle'):
            sub.add_argument('--episodes', type=int, default=8)
            sub.add_argument('--horizon', type=int, default=6)
            sub.add_argument('--seed', type=int, default=96001)
            sub.add_argument('--epsilon', type=float, default=0.15)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'train':
        config = LanguageTrainingConfig(steps=args.steps, batch_size=args.batch_size,
                                        learning_rate=args.learning_rate, rank=args.rank)
        print(train(args.base, args.selection, args.conversations, args.output, config,
                    device=args.device, initialize=args.initialize, resume=args.resume,
                    native_checkpoint=args.native_checkpoint))
        return
    model, payload = load_checkpoint(args.checkpoint, args.base, device=args.device)
    identity = checkpoint_identity(args.checkpoint)
    if args.command == 'evaluate':
        args.output.mkdir(parents=True, exist_ok=True)
        evaluate(model, args.selection, args.validation_conversations, args.output / 'evaluation.json',
                 training_history=payload['training_history'])
        return
    if args.command in ('chat', 'act'):
        session = LanguageSession(model, checkpoint_id=identity)
        if args.session:
            session.restore(args.session)
        picture = waveform = None
        rate = 16000
        if args.image:
            with Image.open(args.image) as source:
                picture = torch.from_numpy(np.array(source.convert('RGB'), dtype=np.float32) / 255)
        if args.audio:
            waveform, rate = read_audio(args.audio)
        feedback = None if args.feedback is None else torch.tensor(args.feedback)
        observation = MultimodalObservation(args.text, picture, waveform, rate, args.previous_action, feedback)
        args.output.mkdir(parents=True, exist_ok=True)
        if args.command == 'chat':
            native = dataclasses.replace(observation, text='') if any(
                item is not None for item in (picture, waveform, args.previous_action, feedback)
            ) else None
            answer = session.reply(args.text, observation=native, max_new_tokens=args.max_new_tokens)
            (args.output / 'response.json').write_text(json.dumps({'text': answer}, ensure_ascii=False, indent=2) + '\n')
            print(answer)
        else:
            decision = session.act(observation, max_text_bytes=args.max_text_bytes)
            print(json.dumps(save_decision(args.output, decision, sample_rate=rate), ensure_ascii=False))
        session.save(args.output / 'session.pt')
        return
    if args.episodes < 1 or args.horizon < 2:
        parser.error('episodes must be positive and horizon at least two')
    excluded = set()
    if args.command == 'cycle':
        excluded = {row['world_id'] for selection in args.selection
                    for row in json.loads(selection.read_text())['episodes'] if row['split'] != 'train'}
    paths, reports, candidate = [], [], args.seed
    while len(paths) < args.episodes:
        seed, candidate = candidate, candidate + 1
        if MultimodalNavigationWorld(seed, horizon=args.horizon).world_id in excluded:
            continue
        _, report = rollout(model, checkpoint_id=identity, seed=seed, horizon=args.horizon, epsilon=args.epsilon,
                            source_root=args.output / 'episodes', display_root=args.output / f'replay-{len(paths):03d}')
        paths.append(Path(report['source_path']))
        reports.append(report)
    selection = write_selection(args.output, {'train': paths})
    (args.output / 'rollouts.json').write_text(json.dumps({'episodes': reports}, ensure_ascii=False, indent=2) + '\n')
    if args.command == 'cycle':
        config = dataclasses.replace(LanguageTrainingConfig(**payload['config']),
                                     steps=args.steps, batch_size=args.batch_size)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        print(train(args.base, [*args.selection, selection], args.conversations, args.output / 'learning',
                    config, device=args.device, initialize=args.checkpoint))
    else:
        print(selection)


if __name__ == '__main__':
    main()
