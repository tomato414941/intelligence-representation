"""Continue one native core with causal text pretraining, then instruction tuning."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from intrep.problems.language_agent.evaluation import evaluate
from intrep.problems.language_agent.training import (
    LanguageTrainingConfig,
    load_checkpoint,
    train,
)
from intrep.sources.language.conversations import (
    check_conversation_split,
    load_conversations,
)
from intrep.sources.language.pretraining import load_pretraining, sample_blocks


@torch.no_grad()
def measure(checkpoint, corpus, selection, validation, output, device):
    model, payload = load_checkpoint(checkpoint, device=device)
    model.eval()
    splits, _ = load_pretraining(corpus)
    generator = torch.Generator().manual_seed(7001)
    losses = [float(model.pretraining_loss(sample_blocks(splits['validation'], 8, 384, generator))) for _ in range(8)]
    documents = json.loads((corpus / 'validation-documents.json').read_text())[:4]
    completions = []
    for document in documents:
        prefix = document[:80]
        completions.append({'prefix': prefix, 'completion': model.complete(prefix, max_new_tokens=256)})
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate(model, selection, validation, output / 'evaluation.json', training_history=payload['training_history'])
    result['pretraining_validation_loss'] = sum(losses) / len(losses)
    result['heldout_completions'] = completions
    (output / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'stage': output.name, 'pretraining_validation_loss': result['pretraining_validation_loss'],
                      'native_action_accuracy': result['native_action_accuracy']}), flush=True)
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--conversations', type=Path, required=True)
    parser.add_argument('--validation-conversations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pretraining-steps', type=int, default=4000)
    parser.add_argument('--instruction-steps', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    check_conversation_split(load_conversations(args.conversations), load_conversations(args.validation_conversations))
    measure(args.checkpoint, args.corpus, args.selection, args.validation_conversations, args.output / 'before', args.device)
    config = LanguageTrainingConfig(steps=args.pretraining_steps, batch_size=2, learning_rate=0.0001,
                                    language_weight=0, native_interval=4, warmup_steps=100,
                                    decay_steps=args.pretraining_steps, beta2=0.95, weight_decay=0.1)
    pretrained = train([args.selection], None, args.output / 'pretraining', config,
                       device=args.device, initialize=args.checkpoint, corpus=args.corpus)
    measure(pretrained, args.corpus, args.selection, args.validation_conversations, args.output / 'after-pretraining', args.device)
    config = LanguageTrainingConfig(steps=args.instruction_steps, batch_size=2, learning_rate=0.0001,
                                    text_weight=0.25, native_interval=1, warmup_steps=50,
                                    decay_steps=args.instruction_steps, beta2=0.95, weight_decay=0.1)
    final = train([args.selection], args.conversations, args.output / 'instruction', config,
                  device=args.device, initialize=pretrained, corpus=args.corpus)
    measure(final, args.corpus, args.selection, args.validation_conversations, args.output / 'after-instruction', args.device)


if __name__ == '__main__':
    main()
