from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MAX_ACTIVE_PLAYERS = 8
SUPPORTED_KINDS = {"checkpoint", "usi_engine"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate the local shogi player registry.")
    parser.add_argument("--registry", type=Path, default=Path("data/shogi/player-registry.json"))
    args = parser.parse_args(argv)

    result = validate_shogi_player_registry(args.registry)
    print(json.dumps(result, indent=2))


def validate_shogi_player_registry(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    max_active_players = int(payload.get("max_active_players", DEFAULT_MAX_ACTIVE_PLAYERS))
    players = payload.get("players")
    if not isinstance(players, list):
        raise ValueError("players must be a list")
    if len(players) > max_active_players:
        raise ValueError(f"active players must be {max_active_players} or fewer")

    seen_ids: set[str] = set()
    kind_counts: dict[str, int] = {}
    for player in players:
        if not isinstance(player, dict):
            raise ValueError("player must be an object")
        player_id = str(player["id"])
        if player_id in seen_ids:
            raise ValueError(f"duplicate player id: {player_id}")
        seen_ids.add(player_id)
        kind = str(player["kind"])
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported player kind: {kind}")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if kind == "checkpoint":
            _require_path(player, "checkpoint")
        if kind == "usi_engine":
            _require_path(player, "command")
            if "go_command" not in player:
                raise ValueError(f"usi_engine player requires go_command: {player_id}")

    return {
        "registry": str(path),
        "max_active_players": max_active_players,
        "player_count": len(players),
        "kind_counts": dict(sorted(kind_counts.items())),
    }


def _require_path(player: dict[str, object], key: str) -> None:
    player_id = str(player["id"])
    value = player.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{player_id} requires {key}")
    if not Path(value).exists():
        raise ValueError(f"{player_id} {key} does not exist: {value}")


if __name__ == "__main__":
    main()
