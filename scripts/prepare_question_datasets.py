"""Download complete FSDD, UCI HAR and labeled BoolQ releases with provenance."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
import urllib.request
import wave
import zipfile
from pathlib import Path

import numpy as np

FSDD_REVISION = "26eb9aaf76e81b692f806f9140c2d2777410d7a1"
HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
HAR_CHANNELS = [f"{kind}_{axis}" for kind in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"]


def identity(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def download(url, path):
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "intrep-dataset-preparation/1"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        temporary.replace(path)
    return {"url": url, **identity(path)}


def prepare_fsdd(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "release.tar.gz"
    provenance = download(f"https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/tar.gz/{FSDD_REVISION}", archive)
    recordings = root / "recordings"
    recordings.mkdir(exist_ok=True)
    rows = []
    with tarfile.open(archive) as source:
        for member in source:
            path = Path(member.name)
            if member.isfile() and path.parent.name == "recordings" and path.suffix == ".wav":
                name = path.name
                digit, speaker, index = name.removesuffix(".wav").split("_")
                content = source.extractfile(member).read()
                with wave.open(io.BytesIO(content)) as audio:
                    if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != 8000:
                        raise ValueError(f"unexpected FSDD audio encoding: {name}")
                    frames = audio.getnframes()
                (recordings / name).write_bytes(content)
                rows.append({"path": f"recordings/{name}", "label": int(digit), "speaker": speaker,
                             "recording": int(index), "frames": frames, "sample_rate": 8000,
                             "sha256": hashlib.sha256(content).hexdigest()})
    rows.sort(key=lambda row: row["path"])
    speakers = sorted({row["speaker"] for row in rows})
    if len(rows) != 3000 or len(speakers) != 6 or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("the pinned FSDD release population differs")
    splits = {speaker: "train" for speaker in speakers[:-2]}
    splits[speakers[-2]], splits[speakers[-1]] = "validation", "test"
    for row in rows:
        row["split"] = splits[row["speaker"]]
    write_json(root / "manifest.json", {"records": rows, "speakers": splits,
               "split_protocol": "custom speaker-disjoint split; not the official recording-index split"})
    write_json(root / "provenance.json", {"source": provenance, "revision": FSDD_REVISION,
               "homepage": "https://github.com/Jakobovski/free-spoken-digit-dataset",
               "license": "CC-BY-SA-4.0", "population": len(rows), "manifest": identity(root / "manifest.json")})
    return {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}


def prepare_har(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "release.zip"
    provenance = download(HAR_URL, archive)
    with zipfile.ZipFile(archive) as outer:
        nested = next((name for name in outer.namelist() if name.endswith("UCI HAR Dataset.zip")), None)
        source = zipfile.ZipFile(io.BytesIO(outer.read(nested))) if nested else outer
        prefix = next(name.removesuffix("train/subject_train.txt") for name in source.namelist()
                      if name.endswith("train/subject_train.txt") and not name.startswith("__MACOSX/"))
        arrays = {}
        for split in ("train", "test"):
            read = lambda name, split=split: io.BytesIO(source.read(prefix + split + "/" + name))
            signals = np.stack([np.loadtxt(read(f"Inertial Signals/{name}_{split}.txt"), dtype=np.float32)
                                for name in HAR_CHANNELS], axis=-1)
            labels = np.loadtxt(read(f"y_{split}.txt"), dtype=np.int64) - 1
            subjects = np.loadtxt(read(f"subject_{split}.txt"), dtype=np.int64)
            arrays[split] = signals, labels, subjects
        if nested:
            source.close()
    if arrays["train"][0].shape != (7352, 128, 9) or arrays["test"][0].shape != (2947, 128, 9):
        raise ValueError("UCI HAR population or inertial signal shape differs")
    if set(arrays["train"][2]) & set(arrays["test"][2]):
        raise ValueError("UCI HAR official subject splits overlap")
    held = sorted(random.Random(9047).sample(sorted(set(map(int, arrays["train"][2]))), 3))
    validation = np.isin(arrays["train"][2], held)
    signals = arrays["train"][0][~validation]
    mean, std = signals.mean(axis=(0, 1)), signals.std(axis=(0, 1)).clip(1e-6)
    counts, subjects = {}, {}
    for split, original, selection in (("train", "train", ~validation), ("validation", "train", validation),
                                       ("test", "test", np.ones(len(arrays["test"][0]), dtype=bool))):
        x, y, group = [array[selection] for array in arrays[original]]
        original_indices = np.flatnonzero(selection)
        np.savez_compressed(root / f"{split}.npz", signals=x, labels=y, subjects=group, original_indices=original_indices)
        counts[split], subjects[split] = len(y), sorted(set(map(int, group)))
    write_json(root / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "channels": HAR_CHANNELS,
               "estimated_from": "all training windows only; validation and test subjects excluded"})
    write_json(root / "provenance.json", {"source": provenance, "homepage": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
               "license": "CC-BY-4.0", "counts": counts, "subjects": subjects, "sample_rate": 50,
               "representation": "nine preprocessed inertial signal channels, 128 samples; not the 561 engineered features",
               "files": [identity(root / name) for name in ("train.npz", "validation.npz", "test.npz", "normalization.json")]})
    return counts


def prepare_boolq(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "release.zip"
    provenance = download("https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip", archive)
    counts = {}
    with zipfile.ZipFile(archive) as source:
        for source_name, split, expected in (("train", "train", 9427), ("val", "validation", 3270)):
            member = next(name for name in source.namelist() if name.endswith(f"/{source_name}.jsonl"))
            rows = [json.loads(line) for line in source.read(member).decode().splitlines()]
            for row in rows:
                row["answer"] = row.pop("label")
            if len(rows) != expected or any(not isinstance(row["answer"], bool) or not row["passage"] or not row["question"] for row in rows):
                raise ValueError("BoolQ labeled population or schema differs")
            (root / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            counts[split] = len(rows)
    write_json(root / "provenance.json", {"source": provenance, "counts": counts,
               "homepage": "https://github.com/google-research-datasets/boolean-questions", "license": "CC-BY-SA-3.0",
               "conversion": "official SuperGLUE v2 distribution; rename label to answer; preserve all passage/question content",
               "files": [identity(root / f"{split}.jsonl") for split in counts],
               "population": "complete official labeled train and development sets; no labeled test set supplied"})
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name, prepare in (("fsdd", prepare_fsdd), ("uci-har", prepare_har), ("boolq", prepare_boolq)):
        print(json.dumps({"dataset": name, "counts": prepare(args.output / name)}), flush=True)


if __name__ == "__main__":
    main()
