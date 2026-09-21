from __future__ import annotations

import copy
import io
import unittest

import torch
from torch import nn

from intrep.learning.joint import JointTrainer
from intrep.representation.assemblies.shared_predictor import SharedPredictor


def small_predictor() -> SharedPredictor:
    model = SharedPredictor(nn.Linear(4, 4), 4)
    model.attach_input("numeric", nn.Linear(3, 4))
    model.attach_output("forecast", nn.Linear(4, 2))
    return model


class SharedPredictorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)

    def test_new_head_names_and_input_formats_do_not_change_core(self):
        model = small_predictor()
        core = model.core
        before = copy.deepcopy(core.state_dict())
        old = model.detach_input("numeric")
        model.attach_input("sensor_events", nn.Embedding(12, 4))
        model.attach_output("tool_result", nn.Linear(4, 7))
        hidden = model(model.encode("sensor_events", torch.tensor([[3, 4, 5]])))
        self.assertEqual(model.decode("tool_result", hidden).shape, (1, 3, 7))
        model.attach_input("numeric", old)
        self.assertIs(model.core, core)
        for name, value in core.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_joint_update_matches_one_weighted_loss_without_dropping_sources(self):
        model = small_predictor()
        expected = copy.deepcopy(model)
        x = torch.randn(1, 3, 3)
        def loss(m, target):
            return (m.decode("forecast", m(m.encode("numeric", x))) - target).square().mean()
        trainer = JointTrainer(model, {"first": 1, "second": 3}, learning_rate=0.1,
                               optimizer="sgd", max_grad_norm=100)
        seen = []
        def source(name, target):
            seen.append(name)
            return loss(model, target)
        expected_metrics = {"first": float(loss(expected, 0).detach()), "second": float(loss(expected, 2).detach())}
        metrics = trainer.step({"first": lambda: source("first", 0), "second": lambda: source("second", 2)})
        for name, value in expected_metrics.items():
            self.assertEqual(metrics[name], value)
        self.assertEqual(metrics["weighted_loss"], (expected_metrics["first"] + 3 * expected_metrics["second"]) / 4)
        (0.25 * loss(expected, 0) + 0.75 * loss(expected, 2)).backward()
        optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
        optimizer.step()
        self.assertEqual(seen, ["first", "second"])
        for actual, wanted in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, wanted)
        # A selected source retains its weight relative to the registered mean.
        optimizer.zero_grad(set_to_none=True)
        (0.5 * loss(expected, 0)).backward()
        optimizer.step()
        trainer.step({"first": lambda: loss(model, 0)})
        for actual, wanted in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, wanted)

    def test_exchange_updates_optimizer_without_resetting_core_history(self):
        model = small_predictor()
        x = torch.randn(1, 2, 3)
        def loss():
            return model.decode("forecast", model(model.encode("numeric", x))).square().mean()
        trainer = JointTrainer(model, {"measurements": 1}, learning_rate=0.01)
        trainer.step({"measurements": loss})
        core_weight = model.core.weight
        core_state = trainer.optimizer.state[core_weight]
        first_moment = core_state["exp_avg"].clone()
        old_weight = model.output_heads["forecast"].weight
        model.attach_output("forecast", nn.Linear(4, 2))
        new_weight = model.output_heads["forecast"].weight
        before = new_weight.clone()
        trainer.synchronize_parameters()
        self.assertIs(trainer.optimizer.state[core_weight], core_state)
        torch.testing.assert_close(core_state["exp_avg"], first_moment, rtol=0, atol=0)
        self.assertNotIn(old_weight, trainer.optimizer.state)
        self.assertEqual(sum(parameter is new_weight for group in trainer.optimizer.param_groups
                             for parameter in group["params"]), 1)
        trainer.step({"measurements": loss})
        self.assertEqual(int(core_state["step"]), 2)
        self.assertEqual(int(trainer.optimizer.state[new_weight]["step"]), 1)
        self.assertFalse(torch.equal(before, new_weight))

    def test_adding_a_source_preserves_the_existing_training_mixture(self):
        model = small_predictor()
        trainer = JointTrainer(model, {"existing": 1}, learning_rate=0.01)
        trainer.add_source("new", 2)
        before = model.core.weight.detach().clone()
        trainer.step({"new": lambda: model.core.weight.square().mean()})
        self.assertFalse(torch.equal(model.core.weight, before))
        self.assertEqual(trainer.weights, {"existing": 1, "new": 2})
        trainer.step({"existing": lambda: model.core.weight.square().mean(),
                      "new": lambda: model.core.weight.abs().mean()})

    def test_invalid_late_loss_does_not_apply_an_earlier_source_update(self):
        model = small_predictor()
        before = copy.deepcopy(model.state_dict())
        trainer = JointTrainer(model, {"good": 1, "bad": 1}, learning_rate=0.01)
        with self.assertRaisesRegex(ValueError, "finite differentiable"):
            trainer.step({"good": lambda: model.core.weight.square().mean(),
                          "bad": lambda: model.core.weight.sum() * float("nan")})
        self.assertEqual(trainer.steps, 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_checkpoint_round_trip_preserves_joint_training_continuation(self):
        model = small_predictor()
        trainer = JointTrainer(model, {"signal": 1}, learning_rate=0.01)
        x = torch.randn(1, 3, 3)
        def loss(m):
            return m.decode("forecast", m(m.encode("numeric", x))).square().mean()
        trainer.step({"signal": lambda: loss(model)})
        stream = io.BytesIO()
        torch.save({"model": model.module_state_dict(), "trainer": trainer.state_dict()}, stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = small_predictor()
        restored.load_module_state_dict(state["model"])
        other = JointTrainer(restored, {"signal": 1}, learning_rate=0.01)
        other.load_state_dict(state["trainer"])
        trainer.step({"signal": lambda: loss(model)})
        other.step({"signal": lambda: loss(restored)})
        for actual, expected in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_nonfinite_loss_or_gradient_cannot_reach_optimizer(self):
        for invalid in ("loss", "gradient"):
            with self.subTest(invalid=invalid):
                model = small_predictor()
                trainer = JointTrainer(model, {"signal": 1}, learning_rate=0.01)
                before = copy.deepcopy(model.state_dict())

                def loss():
                    if invalid == "loss":
                        return model.core.weight.square().mean() + float("nan")
                    return (model.core.weight - model.core.weight.detach()).sum().sqrt()

                with self.assertRaises(ValueError if invalid == "loss" else RuntimeError):
                    trainer.step({"signal": loss})
                self.assertEqual(trainer.steps, 0)
                self.assertFalse(trainer.optimizer.state)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_full_joint_learning_rejects_a_frozen_body(self):
        model = small_predictor()
        model.core.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "remain trainable"):
            JointTrainer(model, {"signal": 1}, learning_rate=0.01)


class LfmSharedCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import Lfm2Config, Lfm2ForCausalLM
        except ImportError as error:
            raise unittest.SkipTest("install the lfm extra for the LFM integration tests") from error
        cls.configuration = Lfm2Config
        cls.model_type = Lfm2ForCausalLM
        torch.set_num_threads(1)

    def native_model(self):
        torch.manual_seed(25)
        return self.model_type(self.configuration(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
            num_attention_heads=2, num_key_value_heads=1, layer_types=["conv", "full_attention", "conv"],
            block_auto_adjust_ff_dim=False, tie_word_embeddings=True, pad_token_id=0,
            bos_token_id=1, eos_token_id=2, attn_implementation="eager",
        ))

    def test_detaching_heads_preserves_pretrained_computation_and_tied_weights(self):
        from intrep.representation.cores.lfm import split_lfm
        native = self.native_model().eval()
        ids = torch.tensor([[1, 7, 3, 9]])
        expected = native(ids, use_cache=False).logits.detach()
        model = split_lfm(native).eval()
        actual = model.decode("text", model(model.encode("text", ids)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(model.core.body.embed_tokens)
        self.assertIs(model.input_heads["text"].weight, model.output_heads["text"].weight)
        self.assertEqual(sum(type(module).__name__ == "Lfm2Model" for module in model.modules()), 1)

    def test_different_head_pairs_update_every_layer_of_the_same_body(self):
        from intrep.representation.cores.lfm import split_lfm
        model = split_lfm(self.native_model())
        model.attach_input("telemetry", nn.Linear(5, 16))
        model.attach_output("continuous_prediction", nn.Linear(16, 3))
        core = model.core
        for name in ("text", "telemetry"):
            model.zero_grad(set_to_none=True)
            inputs = torch.tensor([[1, 3, 8, 2]]) if name == "text" else torch.randn(1, 4, 5)
            output = "text" if name == "text" else "continuous_prediction"
            model.decode(output, model(model.encode(name, inputs))).square().mean().backward()
            for layer in core.body.layers:
                self.assertGreater(float(layer.feed_forward.w2.weight.grad.abs().sum()), 0)
                operator = layer.self_attn.q_proj if layer.is_attention_layer else layer.conv.in_proj
                self.assertGreater(float(operator.weight.grad.abs().sum()), 0)
        self.assertIs(model.core, core)

    def test_causality_and_module_checkpoint_ties(self):
        from intrep.representation.cores.lfm import split_lfm
        model = split_lfm(self.native_model()).eval()
        first = model(model.encode("text", torch.tensor([[1, 3, 7, 8]])))
        second = model(model.encode("text", torch.tensor([[1, 3, 9, 2]])))
        torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)
        saved = copy.deepcopy(model.module_state_dict())
        # A corrupt tied output must be detected before changing the body.
        saved["outputs"]["text"]["weight"] = saved["outputs"]["text"]["weight"].clone() + 1
        with self.assertRaisesRegex(ValueError, "conflicting values"):
            model.load_module_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
