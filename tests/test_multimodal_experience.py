from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import torch

from intrep.experience.multimodal.records import (
    load_episode,
    read_audio,
    save_episode,
    selected_episodes,
    write_audio,
    write_selection,
)
from intrep.problems.multimodal_agent.cli import prepare
from intrep.problems.multimodal_agent.runtime import rollout
from intrep.problems.multimodal_agent.training import (
    MultimodalTrainingConfig,
    load_checkpoint,
    train,
)
from intrep.worlds.gridworld.multimodal import generate_episode
from tests.test_multimodal_agent import tiny_config


class MultimodalExperienceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_media_roundtrip_and_immutable_source(self) -> None:
        episode = generate_episode(211, horizon=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = save_episode(root, episode)
            restored = load_episode(path)
            self.assertEqual(episode.actions, restored.actions)
            for before, after in zip(episode.observations, restored.observations):
                self.assertEqual(before.text, after.text)
                torch.testing.assert_close(before.image, after.image, atol=1/255, rtol=0)
                torch.testing.assert_close(before.audio, after.audio, atol=5e-5, rtol=0)
            with self.assertRaises(FileExistsError):
                save_episode(root, episode)
            write_audio(root / "test.wav", torch.sin(torch.arange(321).float()), 8000)
            wave, rate = read_audio(root / "test.wav")
            self.assertEqual((len(wave), rate), (321, 8000))

    def test_selection_detects_changed_media_and_cross_split_worlds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = generate_episode(212, horizon=2)
            first = save_episode(root / "episodes", episode)
            duplicate_world = dataclasses.replace(episode, id="different-record")
            second = save_episode(root / "episodes", duplicate_world)
            with self.assertRaisesRegex(ValueError, "world leakage"):
                write_selection(root, {"train": [first], "test": [second]})
            selection = write_selection(root, {"train": [first]})
            self.assertEqual(len(selected_episodes(selection, "train")), 1)
            audio = first.parent / "observation-0000.wav"
            payload = bytearray(audio.read_bytes())
            payload[-1] ^= 1
            audio.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "changed"):
                selected_episodes(selection, "train")

    def test_teacher_metadata_does_not_enter_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_episode(Path(directory), generate_episode(213, horizon=2))
            original = load_episode(path)
            payload = json.loads(path.read_text())
            payload["targets"]["teacher_actions"] = [4, 4]
            payload["targets"]["answers"] = ["hidden answer", "another hidden answer"]
            payload["provenance"]["private_state"] = "not an input"
            path.write_text(json.dumps(payload))
            changed = load_episode(path)
            for a, b in zip(original.observations, changed.observations):
                self.assertEqual(a.text, b.text)
                torch.testing.assert_close(a.image, b.image)
                torch.testing.assert_close(a.audio, b.audio)

    def test_prepare_splits_worlds_and_resume_restores_replay_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = prepare(root / "source", train_count=4, validation_count=1, test_count=1, seed=220, horizon=2)
            train_data = selected_episodes(selection, "train")
            test_data = selected_episodes(selection, "test")
            self.assertFalse({e.world_id for e in train_data} & {e.world_id for e in test_data})
            config = MultimodalTrainingConfig(model=tiny_config(), steps=1, batch_size=2, seed=17, warmup_steps=0)
            train([selection], root / "resumed", config, device="cpu")
            two_steps = dataclasses.replace(config, steps=2)
            resumed = train([selection], root / "resumed", two_steps, device="cpu", resume=True)
            straight = train([selection], root / "straight", two_steps, device="cpu")
            left, payload = load_checkpoint(resumed)
            right, _ = load_checkpoint(straight)
            self.assertEqual(payload["step"], 2)
            for name, parameter in left.state_dict().items():
                torch.testing.assert_close(parameter, right.state_dict()[name], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "evict"):
                train([selection], root / "small-replay", dataclasses.replace(config, replay_capacity=2), device="cpu")

    def test_actor_records_executed_actions_without_teacher_targets(self) -> None:
        from intrep.representation.assemblies.multimodal_agent import (
            MultimodalAgentModel,
        )

        model = MultimodalAgentModel(tiny_config())
        with torch.no_grad():
            model.action_output.weight.zero_()
            model.action_output.bias.fill_(-10)
            model.action_output.bias[4] = 10
            model.text_output.weight.zero_()
            model.text_output.bias.fill_(-10)
            model.text_output.bias[258] = 10
        with tempfile.TemporaryDirectory() as directory:
            episode, report = rollout(model, checkpoint_id="fixed-actor", seed=250, source_root=Path(directory), horizon=3)
            self.assertEqual(episode.actions, [4, 4, 4])
            self.assertEqual(episode.teacher_actions, [])
            self.assertEqual(episode.answers, [])
            restored = load_episode(Path(report["source_path"]))
            self.assertEqual(restored.actions, episode.actions)
            self.assertEqual(len(restored.observations), 4)
            self.assertEqual(restored.observations[-1].feedback[2], 1)

    def test_collected_experience_is_mixed_into_learning_and_can_resume(self) -> None:
        from intrep.problems.multimodal_agent.online import collect_and_learn

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = prepare(root / "source", train_count=2, validation_count=1, test_count=1, seed=270, horizon=2)
            config = MultimodalTrainingConfig(model=tiny_config(), steps=1, batch_size=2, warmup_steps=0)
            initial = train([selection], root / "initial", config, device="cpu")
            learned = collect_and_learn(initial, [selection], root / "cycle", rounds=1, episodes_per_round=2,
                                        learning_steps=1, horizon=2, seed=900, device="cpu")
            before, _ = load_checkpoint(initial)
            after, payload = load_checkpoint(learned)
            self.assertEqual(len(payload["sources"]), 2)
            self.assertEqual(len(payload["sources"][1]["episode_ids"]), 2)
            self.assertTrue(any(not torch.equal(value, after.state_dict()[key]) for key, value in before.state_dict().items()))
            replay_selection = root / "cycle" / "round-000" / "selection.json"
            for episode in selected_episodes(replay_selection, "train"):
                self.assertEqual(episode.teacher_actions, [])
                self.assertEqual(episode.answers, [])
            resumed_config = MultimodalTrainingConfig(**{**payload["config"], "model": after.config, "steps": 2})
            resumed = train([selection, replay_selection], learned.parent, resumed_config, device="cpu", resume=True)
            self.assertEqual(load_checkpoint(resumed)[1]["step"], 2)


if __name__ == "__main__":
    unittest.main()
