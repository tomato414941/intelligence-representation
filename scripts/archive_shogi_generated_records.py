from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.domains.shogi.generated_record_archive import archive_shogi_generated_records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Archive generated shogi game records for later selection.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--record-set-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/shogi/records/generated"))
    parser.add_argument("--source-run")
    parser.add_argument("--generation-method")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = archive_shogi_generated_records(
        input_path=args.input,
        output_root=args.output_root,
        record_set_id=args.record_set_id,
        source_run=args.source_run,
        generation_method=args.generation_method,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
