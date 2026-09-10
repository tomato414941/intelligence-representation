from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def render(root: Path, output: Path) -> None:
    episodes = []
    for directory in sorted(root.glob("replay-*")):
        path = directory / "rollout.json"
        if not path.exists():
            continue
        episode = json.loads(path.read_text())
        for row in episode["transitions"]:
            step = directory / f"step-{row['step']:03d}"
            media = {}
            for name in ("observed.png", "predicted.png", "actual.png", "heard.wav", "predicted.wav", "actual.wav"):
                mime = "image/png" if name.endswith(".png") else "audio/wav"
                media[name] = f"data:{mime};base64," + base64.b64encode((step / name).read_bytes()).decode()
            row["media"] = media
        episodes.append(episode)
    if not episodes:
        raise ValueError("no saved actor replays found")
    template = Path(__file__).with_name("multimodal_replay.html").read_text()
    payload = json.dumps({"episodes": episodes}, ensure_ascii=False).replace("<", "\\u003c")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REPLAY_DATA__", payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render real multimodal actor traces as a standalone replay")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.input, args.output)


if __name__ == "__main__":
    main()
