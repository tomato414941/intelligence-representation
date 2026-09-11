"""Joint training with before/after checks on conversations and held-out native experience."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path

import torch

from intrep.problems.language_agent.evaluation import evaluate
from intrep.problems.language_agent.training import (
    LanguageTrainingConfig,
    load_checkpoint,
    read_native_base,
    train,
)
from intrep.representation.assemblies.language_agent import LanguageAgentModel
from intrep.sources.language.conversations import (
    check_conversation_split,
    load_conversations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--native-checkpoint', type=Path, required=True)
    parser.add_argument('--conversations', type=Path, required=True)
    parser.add_argument('--validation-conversations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--skip-evaluation', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(41)
    args.output.mkdir(parents=True, exist_ok=True)
    check_conversation_split(load_conversations(args.conversations), load_conversations(args.validation_conversations))
    if not args.skip_evaluation:
        model = LanguageAgentModel.from_pretrained(args.base, read_native_base(args.native_checkpoint), device=args.device)
        evaluate(model, args.selection, args.validation_conversations, args.output / 'before.json')
        del model
        gc.collect()
        torch.cuda.empty_cache()
    config = LanguageTrainingConfig(steps=args.steps, batch_size=args.batch_size)
    path = train(args.base, [args.selection], args.conversations, args.output / 'learning', config, device=args.device,
                 native_checkpoint=args.native_checkpoint)
    gc.collect()
    torch.cuda.empty_cache()
    if not args.skip_evaluation:
        model, payload = load_checkpoint(path, args.base, device=args.device)
        evaluate(model, args.selection, args.validation_conversations, args.output / 'after.json',
                 training_history=payload['training_history'])
    (args.output / 'measurement.json').write_text(json.dumps({
        'config': dataclasses.asdict(config), 'max_cuda_memory_allocated': torch.cuda.max_memory_allocated(),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
