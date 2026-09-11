from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from intrep.experience.multimodal.records import selected_episodes
from intrep.learning.replay_buffer import ReplayBuffer
from intrep.problems.multimodal_agent.training import (
    SCHEMA as NATIVE_SCHEMA,
)
from intrep.problems.multimodal_agent.training import (
    MultimodalTrainingConfig,
    episode_loss,
    target_values,
)
from intrep.representation.assemblies.language_agent import LanguageAgentModel
from intrep.representation.assemblies.multimodal_agent import MultimodalAgentConfig
from intrep.sources.language.conversations import (
    conversation_source,
    load_conversations,
)
from intrep.sources.language.pretraining import load_pretraining, sample_blocks

SCHEMA = 'intrep.language_agent_checkpoint.v3'


@dataclass(frozen=True)
class LanguageTrainingConfig:
    steps: int = 300
    batch_size: int = 2
    learning_rate: float = 0.0001
    language_weight: float = 1.0
    seed: int = 41
    target_rate: float = 0.02
    text_batch_size: int = 16
    text_block_bytes: int = 384
    text_weight: float = 1.0
    native_interval: int = 1
    warmup_steps: int = 0
    decay_steps: int = 0
    beta2: float = 0.999
    weight_decay: float = 0.01

    def __post_init__(self):
        if (min(self.steps, self.batch_size, self.text_batch_size, self.text_block_bytes, self.native_interval) < 1 or self.learning_rate <= 0
                or self.language_weight < 0 or self.text_weight <= 0 or not 0 < self.target_rate <= 1
                or min(self.warmup_steps, self.decay_steps, self.weight_decay) < 0 or not 0 < self.beta2 < 1
                or (self.decay_steps and self.decay_steps <= self.warmup_steps)):
            raise ValueError('invalid language agent training configuration')


def read_native_base(path: Path) -> dict:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema_version') != NATIVE_SCHEMA:
        raise ValueError('native initialization requires a learned multimodal agent checkpoint')
    return {'config': payload['config']['model'], 'model': payload['model'],
            'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sources': payload.get('training_history', payload['sources'])}


@contextmanager
def target_parameters(model, target):
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    current = {name: value.detach().clone() for name, value in parameters.items()}
    try:
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(target[name])
        yield
    finally:
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(current[name])


def load_checkpoint(path: Path, *, device: str = 'cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema_version') != SCHEMA:
        raise ValueError('requires a single-core language agent checkpoint (v3)')
    model = LanguageAgentModel(MultimodalAgentConfig(**payload['model_config'])).to(device)
    model.load_state_dict(payload['model'], strict=True)
    return model, payload


def train(selections: list[Path], conversations: Path | None, output: Path,
          config: LanguageTrainingConfig, *, device: str = 'cpu', initialize: Path | None = None,
          resume: bool = False, native_checkpoint: Path | None = None, corpus: Path | None = None) -> Path:
    if initialize is not None and resume:
        raise ValueError('choose initialization or exact resume')
    path = output / 'checkpoint.pt'
    if path.exists() and not resume:
        raise FileExistsError('checkpoint exists; use resume or a new directory')
    examples = load_conversations(conversations) if conversations is not None else []
    if not (examples and config.language_weight > 0) and corpus is None:
        raise ValueError('training requires conversations or a pretraining corpus')
    text_splits, text_source = load_pretraining(corpus) if corpus is not None else (None, None)
    if text_splits is not None and len(text_splits['train']) < config.text_block_bytes:
        raise ValueError('pretraining corpus cannot fill a block')
    episodes, sources, partitions, seen = [], [], {}, set()
    for selection in selections:
        for row in json.loads(selection.read_text())['episodes']:
            if partitions.get(row['world_id'], row['split']) != row['split']:
                raise ValueError('a world crosses training and evaluation splits')
            partitions[row['world_id']] = row['split']
        rows = selected_episodes(selection, 'train')
        if seen.intersection(row.id for row in rows):
            raise ValueError('duplicate selected episode')
        seen.update(row.id for row in rows)
        episodes.extend(rows)
        sources.append({'sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
                        'episode_ids': [row.id for row in rows], 'world_ids': [row.world_id for row in rows]})
    if len(episodes) < config.batch_size or (examples and len(examples) < config.batch_size):
        raise ValueError('both native experience and conversation replay must fill a batch')
    provenance = {'episodes': sources, 'conversations': conversation_source(conversations, examples) if examples else
                  {'group_ids': [], 'conversation_ids': []}}
    if text_source is not None:
        provenance['text'] = {'sha256': text_source['token_sha256'], **text_source['train']}
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed + 1)
    start = 0
    payload = None
    if resume or initialize is not None:
        model, payload = load_checkpoint(path if resume else initialize, device=device)
        inherited = payload.get('training_history', [payload['sources']])
        trained_worlds = {world for source in inherited for group in source['episodes'] for world in group['world_ids']}
        if any(split != 'train' and world in trained_worlds for world, split in partitions.items()):
            raise ValueError('initialization has already trained on an evaluation world')
        if resume:
            if ({key: value for key, value in asdict(LanguageTrainingConfig(**payload['config'])).items() if key != 'steps'}
                    != {key: value for key, value in asdict(config).items() if key != 'steps'}
                    or payload['sources'] != provenance):
                raise ValueError('exact resume may change only the step budget')
            start = payload['step']
            if config.steps < start:
                raise ValueError('step budget precedes saved step')
    else:
        if native_checkpoint is None:
            raise ValueError('new training requires a learned native checkpoint')
        native_base = read_native_base(native_checkpoint)
        model = LanguageAgentModel(MultimodalAgentConfig(**native_base['config'])).to(device)
        model.load_state_dict(native_base['model'], strict=True)
        inherited = [{'episodes': native_base['sources'], 'conversations': {'group_ids': [], 'conversation_ids': []}}]
        trained_worlds = {world for source in native_base['sources'] for world in source['world_ids']}
        if any(split != 'train' and world in trained_worlds for world, split in partitions.items()):
            raise ValueError('native initialization has already trained on an evaluation world')
    if text_source is not None:
        prior_documents = {identity for source in inherited for identity in source.get('text', {}).get('document_hashes', [])}
        if prior_documents.intersection(text_source['validation']['document_hashes']):
            raise ValueError('pretraining validation overlaps inherited training documents')
    history = inherited + ([] if provenance in inherited else [provenance])
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    target = {name: value.detach().clone() for name, value in params.items()}
    optimizer = torch.optim.AdamW(list(params.values()), lr=config.learning_rate, betas=(0.9, config.beta2), weight_decay=config.weight_decay)
    if resume:
        optimizer.load_state_dict(payload['optimizer'])
        target = {name: value.to(params[name].device) for name, value in payload['target'].items()}
        generator.set_state(payload['replay_rng'])
        torch.set_rng_state(payload['torch_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
    actor_episodes = [episode for episode in episodes if episode.provenance.get('actor_checkpoint')]
    retained_episodes = [episode for episode in episodes if not episode.provenance.get('actor_checkpoint')]
    actor_replay = ReplayBuffer(capacity=max(1, len(actor_episodes)))
    actor_replay.extend(actor_episodes)
    retained_replay = ReplayBuffer(capacity=max(1, len(retained_episodes)))
    retained_replay.extend(retained_episodes)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    native_config = MultimodalTrainingConfig(model=model.config, text_weight=0.5)
    model.train()
    started = time.perf_counter()
    for step in range(start + 1, config.steps + 1):
        if (step - 1) % config.native_interval:
            batch = []
        elif not actor_episodes:
            batch = retained_replay.sample(config.batch_size, generator=generator)
        elif not retained_episodes:
            batch = actor_replay.sample(config.batch_size, generator=generator)
        elif config.batch_size == 1:
            replay = actor_replay if step % 2 else retained_replay
            batch = replay.sample(1, generator=generator)
        else:
            actor_count = min(len(actor_episodes), max(1, config.batch_size // 2))
            actor_count = max(actor_count, config.batch_size - len(retained_episodes))
            batch = (actor_replay.sample(actor_count, generator=generator)
                     + retained_replay.sample(config.batch_size - actor_count, generator=generator))
        # Static text is sampled directly; the dynamic replay buffers hold native experience.
        conversation_batch = ([examples[index] for index in torch.randperm(len(examples), generator=generator)[:config.batch_size]]
                              if examples and config.language_weight else [])
        if config.decay_steps:
            warmup = min(step / max(1, config.warmup_steps), 1.0)
            progress = min(max(0, step - config.warmup_steps) / (config.decay_steps - config.warmup_steps), 1.0)
            scale = warmup * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
            for group in optimizer.param_groups:
                group['lr'] = config.learning_rate * scale
        optimizer.zero_grad(set_to_none=True)
        native_value, language_value, text_value, metrics = 0.0, 0.0, 0.0, {}
        if batch:
            with target_parameters(model, target):
                bootstraps = target_values(model, batch)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                native_loss, metrics = episode_loss(model, batch, native_config, bootstrap_values=bootstraps)
            if not torch.isfinite(native_loss):
                raise RuntimeError('nonfinite native loss')
            native_loss.backward()
            native_value = float(native_loss.detach())
            del native_loss
        if conversation_batch:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                language_loss = model.conversation_loss(conversation_batch) * config.language_weight
            if not torch.isfinite(language_loss):
                raise RuntimeError('nonfinite conversation loss')
            language_loss.backward()
            language_value = float(language_loss.detach())
            del language_loss
        if text_splits is not None:
            blocks = sample_blocks(text_splits['train'], config.text_batch_size, config.text_block_bytes, generator)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                text_loss = model.pretraining_loss(blocks) * config.text_weight
            if not torch.isfinite(text_loss):
                raise RuntimeError('nonfinite pretraining loss')
            text_loss.backward()
            text_value = float(text_loss.detach())
            del text_loss
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('nonfinite shared model gradient')
        optimizer.step()
        with torch.no_grad():
            for name, value in params.items():
                target[name].lerp_(value, config.target_rate)
        row = {'step': step, 'elapsed_seconds': time.perf_counter() - started,
               'actor_episodes': sum(bool(episode.provenance.get('actor_checkpoint')) for episode in batch),
               'native_loss': native_value, 'language_loss': language_value, 'text_loss': text_value,
               'text_tokens': config.text_batch_size * config.text_block_bytes if text_splits is not None else 0,
               'learning_rate': optimizer.param_groups[0]['lr'], **metrics}
        with (output / 'training.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if step == start + 1 or step % 10 == 0 or step == config.steps:
            print(json.dumps(row), flush=True)
        if step % 100 == 0 or step == config.steps:
            state = {'schema_version': SCHEMA, 'model_config': asdict(model.config), 'config': asdict(config), 'step': step,
                     'sources': provenance, 'training_history': history,
                     'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                     'target': {name: value.cpu() for name, value in target.items()}, 'optimizer': optimizer.state_dict(),
                     'replay_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else []}
            temporary = path.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(path)
    return path
