from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from intrep.representation.inputs.multimodal_observation import MultimodalObservation
from intrep.sources.language.conversations import (
    ChatMessage,
    ConversationExample,
    check_conversation_split,
)

try:
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [3 + byte % 29 for byte in text.encode()]

    def apply_chat_template(self, messages, **kwargs):
        return {'input_ids': [1, *self.encode(str(messages)), 1]}

    def decode(self, ids, **kwargs):
        return ' '.join(str(int(token)) for token in ids)


def tiny_native_base():
    from dataclasses import asdict

    from intrep.representation.assemblies.multimodal_agent import (
        MultimodalAgentConfig,
        MultimodalAgentModel,
    )
    torch.manual_seed(17)
    model = MultimodalAgentModel(MultimodalAgentConfig(embedding_dim=16, hidden_dim=32, num_heads=2,
                                                      num_layers=1, memory_tokens=2, audio_chunk_size=4))
    return {"config": asdict(model.config), "model": model.state_dict(), "sources": [], "checkpoint_sha256": "test"}


def tiny_model():
    from intrep.representation.assemblies.language_agent import LanguageAgentModel
    torch.manual_seed(12)
    backbone = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=24, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=3, num_key_value_heads=1, head_dim=8,
        eos_token_id=2, pad_token_id=0, tie_word_embeddings=True,
    ))
    backbone = get_peft_model(backbone, LoraConfig(r=2, lora_alpha=4, task_type='CAUSAL_LM',
                                                target_modules=['q_proj', 'v_proj']))
    return LanguageAgentModel(backbone, TinyTokenizer(), tiny_native_base())


@unittest.skipUnless(HAS_LLM, 'requires llm extra')
class LanguageAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shared_decoder_receives_language_and_native_gradients(self):
        model = tiny_model()
        for task in ("conversation", "action", "forecast"):
            native = task != "conversation"
            model.zero_grad(set_to_none=True)
            if native:
                observation = MultimodalObservation(text='go', image=torch.rand(2, 2, 3), audio=torch.rand(8))
                memory = model.observe([observation])
                prediction = model.predict_outcome(memory, torch.tensor([1]), image_shapes=[(2, 2)], audio_samples=8)
                loss = (model.policy(memory).square().mean() if task == "action" else
                        prediction.images[0].mean() + prediction.audio.square().mean())
            else:
                loss = model.answer_loss([{'role': 'user', 'content': 'hi'}], 'hello')
            loss.backward()
            self.assertTrue(any('lora_' in name and value.grad is not None and value.grad.abs().sum() > 0
                                for name, value in model.named_parameters()))
            self.assertTrue(all(value.grad is None for name, value in model.core.named_parameters() if 'lora_' not in name))
            if native:
                for layer in (model.native_to_language, model.language_to_native):
                    self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_native_initialization_preserves_learned_operations(self):
        from intrep.representation.assemblies.multimodal_agent import (
            MultimodalAgentConfig,
            MultimodalAgentModel,
        )
        model = tiny_model().eval()
        native = MultimodalAgentModel(MultimodalAgentConfig(**model.native_base['config'])).eval()
        native.load_state_dict(model.native_base['model'])
        with torch.no_grad():
            model.native_gain.zero_()
            observation = MultimodalObservation(text='blue', image=torch.rand(3, 3, 3), audio=torch.rand(8))
            actual = model.observe([observation])
            expected = native.observe([observation])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(model.policy(actual), native.policy(expected), rtol=0, atol=0)
            first = model.predict_outcome(actual, torch.tensor([1]), image_shapes=[(3, 3)], audio_samples=8)
            second = native.predict_outcome(expected, torch.tensor([1]), image_shapes=[(3, 3)], audio_samples=8)
            torch.testing.assert_close(first.images[0], second.images[0], rtol=0, atol=0)
            torch.testing.assert_close(first.audio, second.audio, rtol=0, atol=0)

    def test_answer_shift_and_mask_match_direct_forward(self):
        model = tiny_model().eval()
        messages = [{'role': 'user', 'content': 'hi'}]
        prompt = model.prompt_ids(messages)
        target = model._ids('ok') + [2]
        ids = torch.tensor([prompt + target])
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        expected = model.core(input_ids=ids, labels=labels).loss
        torch.testing.assert_close(model.answer_loss(messages, 'ok'), expected)

    def test_memory_affects_chat_and_forecast_does_not_write_it(self):
        model = tiny_model().eval()
        memory = model.observe([MultimodalObservation(text='left', image=torch.rand(2, 2, 3))])
        before = memory.clone()
        first = model.answer_loss([{'role': 'user', 'content': 'where'}], 'left', memory)
        second = model.answer_loss([{'role': 'user', 'content': 'where'}], 'left', memory * 0)
        self.assertNotEqual(float(first.detach()), float(second.detach()))
        model.predict_outcome(memory, torch.tensor([0]), image_shapes=[(2, 2)], audio_samples=8)
        torch.testing.assert_close(memory, before)
        with patch.object(model, 'policy', side_effect=AssertionError('chat must not act')):
            self.assertIsInstance(model.chat([{'role': 'user', 'content': 'hi'}], max_new_tokens=2), str)
            self.assertIsInstance(model.chat([{'role': 'user', 'content': 'hi'}], memory=memory, max_new_tokens=2), str)

    def test_delta_roundtrip_excludes_frozen_backbone(self):
        model = tiny_model()
        state = model.learned_state()
        self.assertTrue(all('core.' not in name or 'lora_' in name for name in state))
        other = tiny_model()
        other.restore_learned_state(state)
        for key, value in other.learned_state().items():
            torch.testing.assert_close(value, state[key])
        with self.assertRaises(ValueError):
            other.restore_learned_state({})


class ConversationSourceTests(unittest.TestCase):
    def test_tree_split_rejects_related_conversations(self):
        messages = (ChatMessage('user', 'hi'), ChatMessage('assistant', 'hello'))
        with self.assertRaises(ValueError):
            check_conversation_split([ConversationExample('a', 'tree', messages, 'source')],
                                     [ConversationExample('b', 'tree', messages, 'source')])


@unittest.skipUnless(HAS_LLM, 'requires llm extra')
class LanguageTrainingTests(unittest.TestCase):
    def test_exact_resume_and_joint_replay(self):
        import dataclasses
        import json
        import tempfile
        from pathlib import Path

        from intrep.experience.multimodal.records import save_episode, write_selection
        from intrep.problems.language_agent.training import (
            LanguageTrainingConfig,
            train,
        )
        from intrep.worlds.gridworld.multimodal import generate_episode

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / 'base'
            base.mkdir()
            paths = [save_episode(root / 'data' / 'episodes', generate_episode(seed, size=(3, 3), horizon=2))
                     for seed in range(231, 239)]
            actor = generate_episode(239, size=(3, 3), horizon=2)
            actor.teacher_actions, actor.answers = [], []
            actor.provenance['actor_checkpoint'] = 'test-actor'
            paths.append(save_episode(root / 'data' / 'episodes', actor))
            selection = write_selection(root / 'data', {'train': paths})
            conversations = root / 'chat.jsonl'
            conversations.write_text(''.join(json.dumps({'id': f'chat{index}', 'group_id': f'tree{index}', 'source': 'test',
                                                         'messages': [{'role': 'user', 'content': 'hi'},
                                                                      {'role': 'assistant', 'content': 'hello'}]}) + '\n'
                                             for index in (1, 2)))
            config = LanguageTrainingConfig(steps=2, batch_size=2, rank=2)
            with patch('intrep.problems.language_agent.training.LanguageAgentModel.from_pretrained',
                       side_effect=lambda *args, **kwargs: tiny_model()), patch(
                           'intrep.problems.language_agent.training.read_native_base', return_value=tiny_native_base()):
                full = train(base, [selection], conversations, root / 'full', config, native_checkpoint=root / 'native.pt')
                train(base, [selection], conversations, root / 'resumed', dataclasses.replace(config, steps=1), native_checkpoint=root / 'native.pt')
                resumed = train(base, [selection], conversations, root / 'resumed', config, resume=True)
            expected = torch.load(full, weights_only=True)
            actual = torch.load(resumed, weights_only=True)
            self.assertEqual(expected['step'], 2)
            for key, value in expected['model'].items():
                torch.testing.assert_close(value, actual['model'][key], rtol=0, atol=0)
            self.assertEqual(expected['sources']['conversations']['conversation_ids'], ['chat1', 'chat2'])
            sampled = [json.loads(line)['actor_episodes'] for line in (root / 'full' / 'training.jsonl').read_text().splitlines()]
            self.assertEqual(sampled, [1, 1])
            self.assertTrue(any(not torch.equal(value, tiny_model().learned_state()[key])
                                for key, value in expected['model'].items() if 'lora_' in key))

    def test_session_roundtrip_preserves_conversation_and_world_memory(self):
        import tempfile
        from pathlib import Path

        from intrep.problems.language_agent.runtime import LanguageSession

        model = tiny_model()
        session = LanguageSession(model, checkpoint_id='tiny')
        initial_memory = session.memory.clone()
        with patch.object(model, 'chat', return_value='hello') as chat, patch.object(
            model, 'policy', side_effect=AssertionError('chat must not act')
        ):
            self.assertEqual(session.reply('hi'), 'hello')
            self.assertIsNone(chat.call_args.kwargs['memory'])
            self.assertEqual(session.step, 0)
            torch.testing.assert_close(session.memory, initial_memory)
            session.reply('what do you see', observation=MultimodalObservation(image=torch.rand(2, 2, 3)))
            self.assertIsNotNone(chat.call_args.kwargs['memory'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'session.pt'
            session.save(path)
            other = LanguageSession(model, checkpoint_id='tiny')
            other.restore(path)
            self.assertEqual(other.messages, session.messages)
            self.assertTrue(other.has_world_memory)
            torch.testing.assert_close(other.memory, session.memory)
            other.reset()
            self.assertFalse(other.messages)
            self.assertFalse(other.has_world_memory)
