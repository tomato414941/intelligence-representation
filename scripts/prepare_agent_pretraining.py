"""Prepare document-disjoint byte streams from existing TinyStories text files."""
from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from pathlib import Path

import numpy as np

from intrep.representation.inputs.multimodal_observation import EOS


def documents(path):
    lines = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip() == '<|endoftext|>':
                text = ''.join(lines).strip()
                if text:
                    yield text
                lines = []
            else:
                lines.append(line)
    if lines:
        yield ''.join(lines).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-documents', type=int, default=32768)
    parser.add_argument('--validation-documents', type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    streams, provenance, seen = {}, {}, set()
    for split, path, limit in [('validation', args.validation, args.validation_documents),
                                ('train', args.train, args.train_documents)]:
        tokens, identities, selected = array('H'), [], []
        for document in documents(path):
            encoded = document.encode()
            identity = hashlib.sha256(encoded).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            identities.append(identity)
            tokens.extend(encoded)
            tokens.append(EOS)
            if split == 'validation':
                selected.append(document)
            if len(identities) >= limit:
                break
        if not tokens:
            raise ValueError('empty corpus split')
        streams[split] = np.asarray(tokens, dtype=np.uint16)
        provenance[split] = {'document_hashes': identities, 'tokens': len(tokens), 'documents': len(identities),
                             'input_filename': path.name}
        if split == 'validation':
            (args.output / 'validation-documents.json').write_text(json.dumps(selected, ensure_ascii=False) + '\n')
    np.savez_compressed(args.output / 'tokens.npz', **streams)
    provenance['source'] = 'roneneldan/TinyStories; existing local V2-GPT4 files; synthetic English stories'
    provenance['license'] = 'cdla-sharing-1.0'
    provenance['dataset_url'] = 'https://huggingface.co/datasets/roneneldan/TinyStories'
    provenance['token_sha256'] = hashlib.sha256((args.output / 'tokens.npz').read_bytes()).hexdigest()
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({key: {field: value for field, value in row.items() if field != 'document_hashes'}
                      for key, row in provenance.items() if isinstance(row, dict)}))


if __name__ == '__main__':
    main()
