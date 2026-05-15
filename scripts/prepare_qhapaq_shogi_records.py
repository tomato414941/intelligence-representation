from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import py7zr

from intrep.worlds.shogi.kif_io import load_kif_game_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qhapaq KIF archives as compact shogi source-record JSONL.")
    parser.add_argument("--raw-kiffiles-dir", type=Path, default=Path("data/qhapaq/raw/kiffiles"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/qhapaq/processed"))
    parser.add_argument("--games-file", default="qhapaq_games.jsonl")
    parser.add_argument("--failures-file", default="qhapaq_game_failures.jsonl")
    parser.add_argument("--manifest-file", default="manifest.json")
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = prepare_qhapaq_shogi_records(
        raw_kiffiles_dir=args.raw_kiffiles_dir,
        output_dir=args.output_dir,
        games_file=args.games_file,
        failures_file=args.failures_file,
        manifest_file=args.manifest_file,
        max_archives=args.max_archives,
        max_games=args.max_games,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


def prepare_qhapaq_shogi_records(
    *,
    raw_kiffiles_dir: Path,
    output_dir: Path,
    games_file: str,
    failures_file: str,
    manifest_file: str,
    max_archives: int | None = None,
    max_games: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    archives = sorted(raw_kiffiles_dir.glob("*.7z"))
    if max_archives is not None:
        archives = archives[:max_archives]
    if not archives:
        raise ValueError(f"no Qhapaq .7z archives found under {raw_kiffiles_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    games_path = output_dir / games_file
    failures_path = output_dir / failures_file
    manifest_path = output_dir / manifest_file
    _ensure_writable((games_path, failures_path, manifest_path), overwrite=overwrite)

    games_count = 0
    failures_count = 0
    archives_manifest: list[dict[str, object]] = []

    with (
        games_path.open("w", encoding="utf-8") as games_out,
        failures_path.open("w", encoding="utf-8") as failures_out,
    ):
        for archive in archives:
            archive_games = 0
            archive_failures = 0
            with tempfile.TemporaryDirectory(prefix="intrep-qhapaq-") as directory:
                extract_dir = Path(directory)
                try:
                    with py7zr.SevenZipFile(archive) as archive_file:
                        archive_file.extractall(path=extract_dir)
                except Exception as exc:  # noqa: BLE001
                    archive_failures += 1
                    failures_count += 1
                    _write_jsonl(
                        failures_out,
                        {
                            "archive": str(archive),
                            "kif_path": None,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    archives_manifest.append(_archive_manifest(archive, archive_games, archive_failures))
                    continue

                for kif_path in _iter_kif_paths(extract_dir):
                    if max_games is not None and games_count >= max_games:
                        break
                    try:
                        moves, winner = load_kif_game_record(kif_path)
                    except Exception as exc:  # noqa: BLE001
                        archive_failures += 1
                        failures_count += 1
                        _write_jsonl(
                            failures_out,
                            {
                                "archive": str(archive),
                                "kif_path": str(kif_path.relative_to(extract_dir)),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                        continue
                    _write_jsonl(games_out, {"winner": winner, "moves": list(moves)})
                    archive_games += 1
                    games_count += 1
                archives_manifest.append(_archive_manifest(archive, archive_games, archive_failures))
            if max_games is not None and games_count >= max_games:
                break

    manifest = {
        "schema_version": "intrep.qhapaq_shogi_records.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "raw_kiffiles_dir": str(raw_kiffiles_dir),
        "output_dir": str(output_dir),
        "games_file": games_file,
        "failures_file": failures_file,
        "source_archive_count": len(archives_manifest),
        "game_count": games_count,
        "failure_count": failures_count,
        "max_archives": max_archives,
        "max_games": max_games,
        "archives": archives_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "games": str(games_path),
        "failures": str(failures_path),
        "manifest": str(manifest_path),
        "source_archive_count": len(archives_manifest),
        "game_count": games_count,
        "failure_count": failures_count,
    }


def _iter_kif_paths(root: Path) -> Iterable[Path]:
    yield from sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".kif")


def _write_jsonl(file, payload: dict[str, object]) -> None:
    file.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def _ensure_writable(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"output already exists: {', '.join(str(path) for path in existing)}")


def _archive_manifest(archive: Path, game_count: int, failure_count: int) -> dict[str, object]:
    return {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "game_count": game_count,
        "failure_count": failure_count,
    }


if __name__ == "__main__":
    main()
