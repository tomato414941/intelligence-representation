from __future__ import annotations

import json

import torch

from intrep.experience.multimodal.records import selected_episodes
from intrep.problems.language_agent.runtime import LanguageSession
from intrep.problems.multimodal_agent.training import (
    MultimodalTrainingConfig,
    episode_loss,
)
from intrep.sources.language.conversations import load_conversations

# These prompts are never added to either replay buffer.
HELD_OUT = [
    ['こんにちは。今日は何を手伝えますか。短く答えてください。'],
    ['水を冷凍庫に入れると氷になる理由を、小学生向けに短く説明してください。'],
    ['「昨日、妹と公園で散歩しました」を英語に訳してください。'],
    ['次の案内を一文で要約してください。店は水曜日が定休日です。ほかの日は午前10時から午後7時まで営業しています。オンライン注文は定休日にも受け付けています。'],
    ['私の犬の名前はコハクです。短く返事をしてください。', '犬の名前は何でしたか。'],
    ['集合は土曜日の午前10時です。短く返事をしてください。', '日曜日の午前11時に変更します。短く返事をしてください。', '変更後の集合日時を答えてください。'],
    ['8個のりんごを3人に2個ずつ配りました。残りは何個ですか。'],
    ['Write a Python function that filters the even numbers from a list. Return only code.'],
]


@torch.no_grad()
def evaluate(model, selection, validation, output, *, training_history=()):
    episodes = selected_episodes(selection, 'validation')[:8]
    language = load_conversations(validation)[:16]
    trained_worlds = {world for source in training_history for group in source['episodes'] for world in group['world_ids']}
    trained_trees = {tree for source in training_history for tree in source['conversations']['group_ids']}
    if (trained_worlds.intersection(row.world_id for row in episodes)
            or trained_trees.intersection(row.group_id for row in language)):
        raise ValueError('evaluation overlaps the checkpoint training history')
    model.eval()
    cases = []
    for prompts in HELD_OUT:
        session = LanguageSession(model, checkpoint_id='evaluation')
        for text in prompts:
            session.reply(text, max_new_tokens=192)
        cases.append(session.messages)
    language_loss = float(model.conversation_loss(language))
    records, matches, count = [], 0, 0
    config = MultimodalTrainingConfig(model=model.config)
    for episode in episodes:
        loss, metrics = episode_loss(model, [episode], config)
        memory = model.new_memory()
        texts = []
        for step, observation in enumerate(episode.observations[:-1]):
            memory = model.observe([observation], memory, step=step)
            matches += int(model.policy(memory).argmax()) == episode.teacher_actions[step]
            count += 1
            if step == 0:
                texts = model.generate_text(memory, max_bytes=8)
        records.append({'id': episode.id, 'loss': float(loss), **metrics, 'first_text': texts[0],
                        'expected_first_text': episode.answers[0]})
    result = {'conversations': cases, 'language_validation_loss': language_loss,
              'native_action_accuracy': matches / count, 'native_episodes': records,
              'native_mean_losses': {key: sum(row[key] for row in records) / len(records)
                                     for key in ('loss', 'teacher', 'text', 'image', 'audio', 'feedback')}}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('conversations', 'native_episodes')}), flush=True)
    return result
