"""Build bounded, tree-disjoint language replay from human OpenAssistant conversations."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = 'fdf72ae0827c1cda404aff25b6603abec9e3399b'
SOURCE = f'https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/{REVISION}/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=2048)
    parser.add_argument('--validation-count', type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / 'messages.jsonl.gz'
    urllib.request.urlretrieve(SOURCE + '2023-04-12_oasst_ready.messages.jsonl.gz', archive)
    urllib.request.urlretrieve(SOURCE + 'LICENSE', args.output / 'LICENSE')
    with gzip.open(archive, 'rt') as handle:
        messages = {row['message_id']: row for row in map(json.loads, handle)}
    splits = {'train': [], 'validation': []}
    for row in messages.values():
        if row['role'] != 'assistant' or row.get('deleted') or row.get('lang') not in ('en', 'ja') or row.get('rank') != 0:
            continue
        chain, current = [], row
        while current is not None:
            chain.append(current)
            current = messages.get(current.get('parent_id'))
        chain.reverse()
        if chain[0]['role'] != 'prompter' or any(item.get('deleted') for item in chain):
            continue
        if sum(len(item['text']) for item in chain) > 1200:
            continue
        group = row['message_tree_id']
        split = 'validation' if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'
        limit = args.train_count if split == 'train' else args.validation_count
        if len(splits[split]) >= limit:
            continue
        splits[split].append({'id': row['message_id'], 'group_id': group,
                              'source': f'OpenAssistant/oasst1@{REVISION}',
                              'messages': [{'role': 'user' if item['role'] == 'prompter' else 'assistant',
                                            'content': item['text']} for item in chain]})
    for split, rows in splits.items():
        if not rows:
            raise ValueError('no usable conversations')
        (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    (args.output / 'provenance.json').write_text(json.dumps({
        'dataset': 'OpenAssistant/oasst1', 'revision': REVISION, 'license': 'Apache-2.0',
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'counts': {split: len(rows) for split, rows in splits.items()},
        'filter': 'en/ja, undeleted, assistant rank 0, <=1200 characters per chain, tree-hash split',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
