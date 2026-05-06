from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


class ValidateShogiPlayerRegistryScriptTest(unittest.TestCase):
    def test_validates_checkpoint_and_usi_players(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            engine = root / "engine"
            registry = root / "player-registry.json"
            checkpoint.write_bytes(b"checkpoint")
            engine.write_text("#!/bin/sh\n", encoding="utf-8")
            registry.write_text(
                json.dumps(
                    {
                        "max_active_players": 8,
                        "players": [
                            {"id": "baseline", "kind": "checkpoint", "checkpoint": str(checkpoint)},
                            {
                                "id": "yaneuraou-nodes1",
                                "kind": "usi_engine",
                                "command": str(engine),
                                "go_command": "go nodes 1",
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = module.validate_shogi_player_registry(registry)

        self.assertEqual(result["player_count"], 2)
        self.assertEqual(result["kind_counts"], {"checkpoint": 1, "usi_engine": 1})

    def test_rejects_too_many_active_players(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            registry = root / "player-registry.json"
            checkpoint.write_bytes(b"checkpoint")
            registry.write_text(
                json.dumps(
                    {
                        "max_active_players": 1,
                        "players": [
                            {"id": "a", "kind": "checkpoint", "checkpoint": str(checkpoint)},
                            {"id": "b", "kind": "checkpoint", "checkpoint": str(checkpoint)},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "active players"):
                module.validate_shogi_player_registry(registry)

    def test_rejects_duplicate_player_ids(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            registry = root / "player-registry.json"
            checkpoint.write_bytes(b"checkpoint")
            registry.write_text(
                json.dumps(
                    {
                        "players": [
                            {"id": "same", "kind": "checkpoint", "checkpoint": str(checkpoint)},
                            {"id": "same", "kind": "checkpoint", "checkpoint": str(checkpoint)},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                module.validate_shogi_player_registry(registry)


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_shogi_player_registry.py"
    spec = importlib.util.spec_from_file_location("validate_shogi_player_registry", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
