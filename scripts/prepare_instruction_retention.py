"""Mix complete OASST branches with reproducible bilingual instruction examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from intrep.problems.shared_prediction.streams import file_identity


def instructions(seed, count, *, holdout=False):
    rng = random.Random(seed)
    rows = []
    names = ("Ari", "Bea", "Cora", "Dion", "Ena", "Faye")
    colors = (("amber", "琥珀色"), ("navy", "紺色"), ("silver", "銀色"), ("violet", "紫色"))
    for index in range(count):
        a, b = rng.sample(range(101, 200) if holdout else range(1, 40), 2)
        a, b = max(a, b), min(a, b)
        name = names[index % len(names)]
        color, japanese = colors[index % len(colors)]
        word = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=7))
        if holdout:
            cases = [
                ("arithmetic", f"Give the result of subtracting {b} from {a}. Write a numeral without commentary.",
                 f"{a}から{b}を引いた結果を求め、数値のみを記してください。", str(a - b), str(a - b)),
                ("case_conversion", f"Rewrite the code {word} using capital letters, with no surrounding text.",
                 f"コード {word} を英大文字に変換し、そのコードだけ記してください。", word.upper(), word.upper()),
                ("lookup", f"Directory: {name}={color}; Zed=white. Return the value assigned to {name}, and nothing else.",
                 f"対応表：{name}={japanese}、Zed=白色。{name}に対応する値だけを記してください。", color, japanese),
                ("extraction", f"Read this record: ({word}). Return just the content between the parentheses.",
                 f"記録：（{word}）。丸括弧で囲まれた中身だけを記してください。", word, word),
                ("conditional", f"Dispatch rule: odd IDs go to dock A; even IDs go to dock B. ID={a}. Reply with just A or B.",
                 f"振り分け規則：奇数の番号は窓口A、偶数は窓口Bへ。番号は{a}です。AかBの一文字で記してください。",
                 "A" if a % 2 else "B", "A" if a % 2 else "B"),
                ("quantity", f"A crate holds {a} pieces. After removing {b} pieces, how many remain? Respond with a numeral only.",
                 f"箱に部品が{a}個あります。{b}個取り出したあとの個数を、数値のみで記してください。", str(a - b), str(a - b)),
            ]
        else:
            cases = [
                ("arithmetic", f"Calculate {a} + {b}. Answer with the number alone.",
                 f"{a}と{b}の和を計算してください。答えの数字のみを出力してください。", str(a + b), str(a + b)),
                ("case_conversion", f"Change {word.upper()} to lowercase. Give only the lowercase text.",
                 f"{word.upper()} を英小文字に直してください。変換後の文字列のみを出力してください。", word, word),
                ("lookup", f"{name}'s bag is {color}. Their shoes are black. What color is the bag? Give only the color.",
                 f"{name}のかばんは{japanese}で、靴は黒色です。かばんの色のみを出力してください。", color, japanese),
                ("extraction", f"Copy the text enclosed in square brackets: [{word}]. Return just that text.",
                 f"角括弧内の文字列をそのまま写してください：[{word}]。括弧の中身のみを出力してください。", word, word),
                ("conditional", f"If the count exceeds 20, output high; otherwise output low. Count: {a}.",
                 f"個数が20より大きければ「多い」、そうでなければ「少ない」と答えてください。個数は{a}です。",
                 "high" if a > 20 else "low", "多い" if a > 20 else "少ない"),
                ("summary", f"Summarize in one sentence: {name} bought {b} books. All {b} books were for the library.",
                 f"次の内容を一文にまとめてください：{name}は本を{b}冊買いました。その{b}冊はすべて図書館用でした。",
                 f"{name} bought {b} books for the library.", f"{name}は図書館用に本を{b}冊買いました。"),
            ]
        for family, english_prompt, japanese_prompt, english_answer, japanese_answer in cases:
            for language, prompt, answer in (("en", english_prompt, english_answer), ("ja", japanese_prompt, japanese_answer)):
                rows.append({"id": f"{'holdout' if holdout else 'instruction'}:{family}:{index}:{language}",
                             "family": family, "language": language, "prompt": prompt,
                             "expected": answer, "max_tokens": 48 if family == "summary" else 16})
    # Parameterized templates can coincide; count unique prompts, not replicas.
    unique = {row["prompt"]: row for row in rows}
    return list(unique.values())


def explanations():
    facts = [
        ("Why does a shadow form?", "A shadow forms when an object blocks light.",
         "影ができる理由は何ですか。", "物体が光を遮るため、影ができます。"),
        ("Why can metal feel colder than wood at the same room temperature?", "Metal transfers heat away from your hand faster than wood does.",
         "同じ室温でも金属が木より冷たく感じるのはなぜですか。", "金属は木よりも手の熱を速く奪うためです。"),
        ("Why do plants need light?", "Plants use light energy in photosynthesis to make sugars.",
         "植物に光が必要なのはなぜですか。", "植物は光合成で光のエネルギーを使って糖を作るためです。"),
        ("Why does a ball fall when released near Earth's surface?", "Earth's gravity pulls the ball downward.",
         "地表近くで手を離したボールが落ちるのはなぜですか。", "地球の重力がボールを下へ引くためです。"),
        ("Why does rubbing your hands warm them?", "Friction converts some of the motion's energy into heat.",
         "手をこすると温かくなるのはなぜですか。", "摩擦によって運動のエネルギーの一部が熱に変わるためです。"),
        ("Why can we hear a vibrating string?", "The vibrating string produces pressure waves that travel through the air to our ears.",
         "振動する弦の音が聞こえるのはなぜですか。", "弦の振動が空気中に圧力の波を作り、耳に届くためです。"),
    ]
    rows = []
    for index, (ep, ea, jp, ja) in enumerate(facts):
        for language, prompt, answer in (("en", ep + " Explain in one sentence.", ea), ("ja", jp + "一文で説明してください。", ja)):
            rows.append({"id": f"instruction:explanation:{index}:{language}", "family": "explanation",
                         "language": language, "prompt": prompt, "expected": answer})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--development-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    training = instructions(47219, 1000) + explanations()
    heldout = instructions(91471, 5, holdout=True)
    development = json.loads(args.development_prompts.read_text())
    forbidden = {row["prompt"] for row in development + heldout}
    training = [row for row in training if row["prompt"] not in forbidden]
    original = {}
    counts = {}
    train_prompts = set()
    for split in ("train", "validation"):
        path = args.conversations / f"{split}.jsonl"
        original[split] = file_identity(path)
        rows = [json.loads(line) for line in path.open()]
        counts[f"original_{split}"] = len(rows)
        if split == "train":
            for row in rows:
                train_prompts.update(message["content"] for message in row["messages"] if message["role"] == "user")
            if forbidden & train_prompts:
                raise ValueError("an evaluation prompt already occurs in the original training population")
            rows.extend({"id": row["id"], "group_id": row["id"].rsplit(":", 1)[0],
                         "source": "intrep.bilingual-instructions.v1", "family": row["family"], "language": row["language"],
                         "messages": [{"role": "user", "content": row["prompt"]},
                                      {"role": "assistant", "content": row["expected"]}]} for row in training)
        # Keep all original branches and use the same seeded order in all conditions.
        random.Random(47219).shuffle(rows)
        with (args.output / f"{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
    (args.output / "holdout-prompts.json").write_text(json.dumps(heldout, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(args.conversations / "LICENSE", args.output / "OASST-LICENSE")
    metadata = {"schema": "intrep.instruction-retention-data.v1", "original_files": original,
                "original_provenance": json.loads((args.conversations / "provenance.json").read_text()),
                "counts": counts, "synthetic_instructions": len(training),
                "synthetic_by_language": dict(Counter(row["language"] for row in training)),
                "synthetic_by_family": dict(Counter(row["family"] for row in training)),
                "heldout_prompts": len(heldout), "development_prompts": file_identity(args.development_prompts),
                "heldout_sha256": hashlib.sha256((args.output / "holdout-prompts.json").read_bytes()).hexdigest(),
                "selection": "Every original conversation branch plus unique programmatically authored training instructions; no teacher model.",
                "evaluation": "Existing 90 scored probes are development data. Fresh wording and values are evaluated only before and after training.",
                "limits": "Synthetic templates are narrow; no claim that these examples represent general instruction diversity."}
    (args.output / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
