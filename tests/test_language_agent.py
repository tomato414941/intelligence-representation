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
    from intrep.representation.assemblies.multimodal_agent import MultimodalAgentConfig
    native = tiny_native_base()
    model = LanguageAgentModel(MultimodalAgentConfig(**native['config']))
    model.load_state_dict(native['model'])
    return model


class LanguageAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_one_core_receives_language_action_forecast_and_memory_gradients(self):
        from intrep.representation.cores.transformer import SharedTransformerCore
        model = tiny_model()
        self.assertEqual(sum(isinstance(module, SharedTransformerCore) for module in model.modules()), 1)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(set(model.state_dict()), set(tiny_native_base()['model']))
        for task in ('conversation', 'action', 'forecast'):
            model.zero_grad(set_to_none=True)
            if task == 'conversation':
                loss = model.answer_loss([{'role': 'user', 'content': 'hi'}], 'hello')
            else:
                memory = model.observe([MultimodalObservation(text='go', image=torch.rand(2, 2, 3), audio=torch.rand(8))])
                prediction = model.predict_outcome(memory, torch.tensor([1]), image_shapes=[(2, 2)], audio_samples=8)
                loss = (model.policy(memory).square().mean() if task == 'action' else
                        prediction.images[0].mean() + prediction.audio.square().mean())
            loss.backward()
            for layer in model.core.layers:
                self.assertGreater(float(layer.attention.in_proj_weight.grad.abs().sum()), 0)
                self.assertGreater(float(layer.feed_forward_input.weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.memory_gate.weight.grad.abs().sum()), 0)

    def test_native_initialization_preserves_all_operations(self):
        from intrep.representation.assemblies.multimodal_agent import (
            MultimodalAgentConfig,
            MultimodalAgentModel,
        )
        model = tiny_model().eval()
        native = MultimodalAgentModel(MultimodalAgentConfig(**tiny_native_base()['config'])).eval()
        native.load_state_dict(tiny_native_base()['model'])
        with torch.no_grad():
            observation = MultimodalObservation(text='blue', image=torch.rand(3, 3, 3), audio=torch.rand(8))
            actual, expected = model.observe([observation]), native.observe([observation])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(model.policy(actual), native.policy(expected), rtol=0, atol=0)
            first = model.predict_outcome(actual, torch.tensor([1]), image_shapes=[(3, 3)], audio_samples=8)
            second = native.predict_outcome(expected, torch.tensor([1]), image_shapes=[(3, 3)], audio_samples=8)
            torch.testing.assert_close(first.images[0], second.images[0], rtol=0, atol=0)
            torch.testing.assert_close(first.audio, second.audio, rtol=0, atol=0)
            torch.testing.assert_close(model.text_logits(actual, [[10]])[0], native.text_logits(expected, [[10]])[0])

    def test_answer_shift_mask_and_causality(self):
        from intrep.representation.inputs.multimodal_observation import EOS
        model = tiny_model().eval()
        messages = [{'role': 'user', 'content': 'hi'}]
        prompt = model.prompt_ids(messages)
        memory = model.conversation_memory(prompt)
        target = [111, 107, EOS]
        logits = model.text_logits(memory, [prompt + target[:-1]])[0]
        expected = torch.nn.functional.cross_entropy(logits[len(prompt):], torch.tensor(target))
        torch.testing.assert_close(model.answer_loss(messages, 'ok'), expected)
        changed = model.text_logits(memory, [prompt + [99, 98]])[0]
        torch.testing.assert_close(logits[:len(prompt) + 1], changed[:len(prompt) + 1])

    def test_long_answer_supervises_every_byte_and_eos_once(self):
        from intrep.representation.inputs.multimodal_observation import EOS
        model = tiny_model()
        model.context_bytes = 3
        captured = []
        original = torch.nn.functional.cross_entropy
        def loss(logits, labels, **kwargs):
            captured.extend(labels.tolist())
            return original(logits, labels, **kwargs)
        with patch('intrep.representation.assemblies.language_agent.F.cross_entropy', side_effect=loss):
            model.answer_loss([{'role': 'user', 'content': 'hi'}], 'abcdefgh').backward()
        self.assertEqual(captured, [*b'abcdefgh', EOS])

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
            self.assertIsInstance(model.chat([{'role': 'user', 'content': 'hi'}], memory=memory, max_new_tokens=2), str)


    def test_generation_enforces_utf8_including_budget_and_surrogate_boundaries(self):
        from intrep.representation.inputs.multimodal_observation import (
            EOS,
            TEXT_VOCAB_SIZE,
        )
        model = tiny_model().eval()
        scores = torch.full((1, TEXT_VOCAB_SIZE), -10.0)
        scores[0, 0x80] = 100  # A continuation cannot start a codepoint.
        scores[0, 0xED] = 90
        scores[0, 0xA0] = 110  # ED A0 would encode a surrogate and is forbidden.
        scores[0, 0x9F] = 70
        scores[0, ord('a')] = 60
        scores[0, EOS] = -100
        with patch.object(model, 'text_logits', return_value=[scores]):
            one = model.chat([{'role': 'user', 'content': 'hi'}], max_new_tokens=1)
            three = model.chat([{'role': 'user', 'content': 'hi'}], max_new_tokens=3)
        self.assertEqual(one, 'a')
        self.assertEqual(three.encode('utf-8'), bytes([0xED, 0x80, 0xA0]))


    def test_training_and_generation_prefix_match_across_context_boundaries(self):
        model = tiny_model().eval()
        model.context_bytes = 3
        prompt = model.prompt_ids([{'role': 'user', 'content': 'hi'}])
        target = list(b'abcdefgh')
        memory = model.conversation_memory(prompt)
        for start in range(0, len(target), 3):
            labels = target[start:start + 3]
            prefix = (prompt + target[:start])[-3:]
            logits = model.text_logits(memory, [prefix + labels[:-1]])[0][len(prefix):]
            for offset in range(len(labels)):
                generated_prefix = model._answer_prefix(prompt, target[:start + offset])
                actual = model.text_logits(memory, [generated_prefix])[0][-1]
                torch.testing.assert_close(actual, logits[offset], rtol=1e-5, atol=1e-6)

    def test_pretraining_shift_causality_and_shared_core(self):
        model = tiny_model()
        blocks = torch.tensor([list(b'abcdef'), list(b'ghijkl')])
        with patch.object(model, 'observe', side_effect=AssertionError('targets must not enter memory')):
            loss = model.pretraining_loss(blocks)
        loss.backward()
        self.assertGreater(float(model.core.layers[0].attention.in_proj_weight.grad.abs().sum()), 0)
        logits = model.text_logits(model.new_memory(2), blocks[:, :-1].tolist())
        expected = torch.nn.functional.cross_entropy(torch.cat(logits), blocks.flatten())
        torch.testing.assert_close(loss, expected)
        changed = model.text_logits(model.new_memory(2), [[97, 98, 99, 0, 0], list(b'ghijk')])
        torch.testing.assert_close(logits[0][:4], changed[0][:4])


class ConversationSourceTests(unittest.TestCase):
    def test_tree_split_rejects_related_conversations(self):
        messages = (ChatMessage('user', 'hi'), ChatMessage('assistant', 'hello'))
        with self.assertRaises(ValueError):
            check_conversation_split([ConversationExample('a', 'tree', messages, 'source')],
                                     [ConversationExample('b', 'tree', messages, 'source')])


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
            import hashlib

            import numpy as np
            corpus = root / 'corpus'
            corpus.mkdir()
            np.savez(corpus / 'tokens.npz', train=np.array([*b'abcdefghij'] * 3), validation=np.array([*b'klmnopqr']))
            (corpus / 'provenance.json').write_text(json.dumps({
                'token_sha256': hashlib.sha256((corpus / 'tokens.npz').read_bytes()).hexdigest(),
                'train': {'document_hashes': ['train'], 'tokens': 30},
                'validation': {'document_hashes': ['validation'], 'tokens': 8}}))
            config = LanguageTrainingConfig(steps=2, batch_size=2, text_batch_size=2, text_block_bytes=4,
                                            warmup_steps=1, decay_steps=5, beta2=0.95)
            with patch('intrep.problems.language_agent.training.read_native_base', return_value=tiny_native_base()):
                full = train([selection], conversations, root / 'full', config, native_checkpoint=root / 'native.pt', corpus=corpus)
                train([selection], conversations, root / 'resumed', dataclasses.replace(config, steps=1), native_checkpoint=root / 'native.pt', corpus=corpus)
                resumed = train([selection], conversations, root / 'resumed', config, resume=True, corpus=corpus)
            expected = torch.load(full, weights_only=True)
            actual = torch.load(resumed, weights_only=True)
            self.assertEqual(expected['step'], 2)
            for key, value in expected['model'].items():
                torch.testing.assert_close(value, actual['model'][key], rtol=0, atol=0)
            self.assertEqual(expected['sources']['conversations']['conversation_ids'], ['chat1', 'chat2'])
            sampled = [json.loads(line)['actor_episodes'] for line in (root / 'full' / 'training.jsonl').read_text().splitlines()]
            self.assertEqual(sampled, [1, 1])
            self.assertTrue(any(not torch.equal(value, tiny_model().state_dict()[key])
                                for key, value in expected['model'].items() if key.startswith('core.')))

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
            self.assertIsNotNone(chat.call_args.kwargs['memory'])
            self.assertEqual(session.step, 2)
            self.assertFalse(torch.equal(session.memory, initial_memory))
            session.reply('what do you see', observation=MultimodalObservation(image=torch.rand(2, 2, 3)))
            self.assertIsNotNone(chat.call_args.kwargs['memory'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'session.pt'
            session.save(path)
            other = LanguageSession(model, checkpoint_id='tiny')
            other.restore(path)
            self.assertEqual(other.messages, session.messages)
            torch.testing.assert_close(other.memory, session.memory)
            other.reset()
            self.assertFalse(other.messages)


    def test_failed_reply_restores_common_memory_and_messages(self):
        from intrep.problems.language_agent.runtime import LanguageSession
        model = tiny_model()
        session = LanguageSession(model, checkpoint_id='tiny')
        before = session.memory.clone()
        with (patch.object(model, 'chat', side_effect=RuntimeError('generation failed')),
              self.assertRaises(RuntimeError)):
            session.reply('hi', observation=MultimodalObservation(image=torch.rand(2, 2, 3)))
        torch.testing.assert_close(session.memory, before)
        self.assertEqual(session.step, 0)
        self.assertEqual(session.messages, [])

    def test_rejects_old_two_model_checkpoint(self):
        import tempfile
        from pathlib import Path

        from intrep.problems.language_agent.training import load_checkpoint
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'old.pt'
            torch.save({'schema_version': 'intrep.language_agent_checkpoint.v2'}, path)
            with self.assertRaisesRegex(ValueError, 'single-core'):
                load_checkpoint(path)
