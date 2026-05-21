from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.domains.shogi.info_stats import inspect_shogi_usi_info_jsonl, write_shogi_usi_info_stats_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw USI info lines in shogi game records.")
    parser.add_argument("--games-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path)
    args = parser.parse_args()

    stats = inspect_shogi_usi_info_jsonl(args.games_jsonl)
    if args.metrics_json is not None:
        write_shogi_usi_info_stats_json(args.metrics_json, stats)
    print(json.dumps(stats.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
