from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.worlds.shogi.experience_store import append_shogi_experience_store


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Append shogi game records to an experience store.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=Path("data/shogi/experiences/train"))
    args = parser.parse_args(argv)

    result = append_shogi_experience_store(input_path=args.input, store_dir=args.store)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
