from __future__ import annotations

import copy
import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch

from intrep.problems.multimodal_agent.runtime import AgentSession
from intrep.problems.multimodal_agent.training import (
    MultimodalTrainingConfig,
    episode_loss,
)
from intrep.representation.assemblies.multimodal_agent import (
    MultimodalAgentConfig,
    MultimodalAgentModel,
)
from intrep.representation.inputs.multimodal_observation import (
    MultimodalObservation,
    image_patches,
    patches_to_image,
)
from intrep.worlds.gridworld.multimodal import (
    MultimodalNavigationWorld,
    generate_episode,
)


def tiny_config() -> MultimodalAgentConfig:
    return MultimodalAgentConfig(embedding_dim=16, hidden_dim=32, num_heads=2, num_layers=1,
                                memory_tokens=4, audio_chunk_size=128)


class MultimodalAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        torch.manual_seed(12)
        self.model = MultimodalAgentModel(tiny_config()).eval()

    def test_variable_images_and_padding_match_individual_inference(self) -> None:
        observations = [generate_episode(81, size=(3, 4), horizon=2).observations[0],
                        generate_episode(82, size=(5, 3), horizon=2).observations[0]]
        observations[1] = dataclasses.replace(observations[1], text="Go blue. " * 3, audio=observations[1].audio[:257])
        batched = self.model.observe(observations)
        individual = torch.cat([self.model.observe([observation]) for observation in observations])
        torch.testing.assert_close(batched, individual, atol=1e-5, rtol=1e-5)
        predictions = self.model.predict_outcome(batched, torch.tensor([1, 2]), image_shapes=[(3, 4), (5, 3)], audio_samples=257)
        self.assertEqual([tuple(image.shape) for image in predictions.images], [(3, 4, 3), (5, 3, 3)])
        self.assertEqual(predictions.audio.shape, (2, 257))

    def test_nondivisible_image_patch_roundtrip(self) -> None:
        image = torch.rand(3, 5, 3)
        patches, grid = image_patches(image, 2)
        self.assertEqual(grid, (2, 3))
        torch.testing.assert_close(patches_to_image(patches, (3, 5), 2), image)
        model = MultimodalAgentModel(dataclasses.replace(tiny_config(), image_patch_size=2))
        memory = model.observe([MultimodalObservation(image=image)])
        result = model.predict_outcome(memory, torch.tensor([0]), image_shapes=[(3, 5)], audio_samples=0)
        self.assertEqual(result.images[0].shape, image.shape)
        self.assertEqual(result.audio.shape, (1, 0))

    def test_native_single_modality_inputs_and_output_requests(self) -> None:
        observations = [MultimodalObservation(text="場所を覚える"), MultimodalObservation(image=torch.rand(3, 4, 3)),
                        MultimodalObservation(audio=torch.rand(213) * 2 - 1, sample_rate=8000),
                        MultimodalObservation(previous_action=2), MultimodalObservation(feedback=torch.tensor([1., 0., 0.]))]
        memory = self.model.observe(observations)
        self.assertEqual(memory.shape, (5, 4, 16))
        result = self.model.predict_outcome(memory, torch.tensor([0, 1, 2, 3, 4]), image_shapes=[None] * 5, audio_samples=0)
        self.assertEqual(result.images, [None] * 5)
        self.assertEqual(result.feedback.shape, (5, 3))

    def test_all_inputs_and_memory_receive_gradients_from_later_decision(self) -> None:
        first = MultimodalObservation(text="Remember blue", image=torch.rand(3, 4, 3), audio=torch.rand(256) * 2 - 1)
        previous = self.model.observe([first])
        previous.retain_grad()
        second = MultimodalObservation(image=torch.rand(4, 3, 3), previous_action=1, feedback=torch.tensor([0.1, 0., 0.]))
        current = self.model.observe([second], previous, step=1)
        self.model.policy(current)[0, 0].backward()
        self.assertGreater(float(previous.grad.abs().sum()), 0)
        for name in ("text", "image", "audio", "action", "feedback"):
            gradient = getattr(self.model.observation_input, name).weight.grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0, name)
        self.assertGreater(float(self.model.memory_gate.weight.grad.abs().sum()), 0)

    def test_text_decoder_cannot_see_future_teacher_tokens(self) -> None:
        memory = self.model.observe([MultimodalObservation(text="a")])
        first = self.model.text_logits(memory, [[65, 66, 67]])[0]
        second = self.model.text_logits(memory, [[65, 90, 91]])[0]
        torch.testing.assert_close(first[:2], second[:2], atol=1e-6, rtol=1e-6)
        self.assertGreater(float((first[2:] - second[2:]).detach().abs().max()), 1e-5)

    def test_forecast_requests_do_not_mutate_memory_or_action_decision(self) -> None:
        memory = self.model.observe([MultimodalObservation(text="go red", image=torch.rand(3, 4, 3))])
        saved = memory.detach().clone()
        before = self.model.policy(memory)
        first = self.model.predict_outcome(memory, torch.tensor([0]), image_shapes=[(3, 4)], audio_samples=256)
        second = self.model.predict_outcome(memory, torch.tensor([1]), image_shapes=[(3, 4)], audio_samples=256)
        torch.testing.assert_close(memory, saved)
        torch.testing.assert_close(self.model.policy(memory), before)
        self.assertGreater(float((first.images[0] - second.images[0]).detach().abs().max()), 1e-6)

    def test_replay_loss_uses_real_outcomes_with_a_detached_target_network(self) -> None:
        episodes = [generate_episode(83, size=(3, 3), horizon=2), generate_episode(84, size=(4, 3), horizon=3)]
        target = copy.deepcopy(self.model)
        config = MultimodalTrainingConfig(model=tiny_config(), steps=1, batch_size=2)
        loss, metrics = episode_loss(self.model, episodes, config, target_model=target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))
        for head in ("action_output", "image_output", "audio_output", "feedback_output", "text_output"):
            self.assertGreater(float(getattr(self.model, head).weight.grad.abs().sum()), 0, head)
        self.assertTrue(all(metrics[name] > 0 for name in ("teacher", "value", "text", "image", "audio", "feedback")))

    def test_unlabelled_actor_episode_can_train_from_reward_and_observation(self) -> None:
        episode = generate_episode(85, size=(3, 3), horizon=2)
        episode.teacher_actions, episode.answers = [], []
        loss, metrics = episode_loss(self.model, [episode], MultimodalTrainingConfig(model=tiny_config()),
                                    target_model=copy.deepcopy(self.model))
        loss.backward()
        self.assertEqual(metrics["teacher"], 0)
        self.assertEqual(metrics["text"], 0)
        self.assertGreater(metrics["value"], 0)
        self.assertGreater(float(self.model.action_output.weight.grad.abs().sum()), 0)

    def test_session_restore_reset_and_model_binding(self) -> None:
        session = AgentSession(self.model, checkpoint_id="model-a", seed=3)
        observation = MultimodalObservation(text="remember red")
        session.act(observation, max_text_bytes=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.pt"
            session.save(path)
            restored = AgentSession(self.model, checkpoint_id="model-a", seed=99)
            restored.restore(path)
            a = session.act(observation, epsilon=1, max_text_bytes=2)
            b = restored.act(observation, epsilon=1, max_text_bytes=2)
            self.assertEqual(a.action, b.action)
            torch.testing.assert_close(session.memory, restored.memory)
            with self.assertRaisesRegex(ValueError, "different model"):
                AgentSession(self.model, checkpoint_id="model-b").restore(path)
        session.reset()
        self.assertEqual(session.step, 0)
        torch.testing.assert_close(session.memory, self.model.new_memory())

    def test_world_text_audio_and_image_carry_distinct_information(self) -> None:
        world = MultimodalNavigationWorld(86, size=(3, 4), horizon=4)
        initial = world.observe()
        world.target = 1 - world.target
        other_text = world.observe()
        self.assertNotEqual(initial.text, other_text.text)
        torch.testing.assert_close(initial.image, other_text.image)
        torch.testing.assert_close(initial.audio, other_text.audio)
        world.reversed_controls = not world.reversed_controls
        other_audio = world.observe()
        self.assertEqual(other_text.text, other_audio.text)
        torch.testing.assert_close(other_text.image, other_audio.image)
        self.assertFalse(torch.equal(other_text.audio, other_audio.audio))

    def test_invalid_observations_are_rejected(self) -> None:
        invalid = [MultimodalObservation(), MultimodalObservation(audio=torch.zeros(0)),
                   MultimodalObservation(image=torch.ones(3, 3, 3) * 2),
                   MultimodalObservation(previous_action=5), MultimodalObservation(feedback=torch.zeros(2))]
        for observation in invalid:
            with self.assertRaises(ValueError):
                self.model.observe([observation])


if __name__ == "__main__":
    unittest.main()
