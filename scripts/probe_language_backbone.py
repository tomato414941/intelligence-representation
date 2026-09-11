"""Inspect real language responses before adopting a pretrained backbone."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CASES = [
    ["こんにちは。日本語で一言、挨拶してください。"],
    ["雨が降る仕組みを、小学生にも分かる日本語で二文で説明してください。"],
    ["次の文を英語に訳してください。明日は友達と図書館へ行きます。"],
    ["次を一文に要約してください。図書館は月曜日に休館します。火曜日から日曜日は朝9時から夕方6時まで開館します。返却ポストは休館日も使えます。"],
    ['赤、青、緑を英語に訳し、日本語をキー、英語を値とするJSONオブジェクトだけを返してください。'],
    ["17足す28はいくつですか。答えの数字だけ返してください。"],
    ["私の猫の名前はムギです。", "猫の名前を覚えていますか。名前だけ答えてください。"],
    ["会議は木曜日の午後3時です。", "訂正します。金曜日の午後4時に変更されました。", "現在の会議の曜日と時刻を答えてください。"],
    ["AはBより小さく、BはCより小さいです。一番大きいものを一文字で答えてください。"],
    ["Write a Python function that returns the squares of all numbers in a list. Return only code."],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--export-base", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(41)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True) if args.quantize else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        device_map={"": args.device}, quantization_config=quantization,
    ).eval()
    records = []
    for index, prompts in enumerate(CASES):
        messages = []
        started = time.perf_counter()
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False, return_dict=True, return_tensors="pt")
            inputs = {name: value.to(args.device) for name, value in inputs.items()}
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": answer})
        record = {"case": index, "messages": messages, "seconds": time.perf_counter() - started}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    report = {"base_model": args.model, "revision": args.revision,
              "quantized_nf4": args.quantize, "cases": records,
              "purpose": "Development readiness examples, not held-out training acceptance scores.",
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes")}}
    (args.output / "readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.export_base:
        destination = args.output / "base"
        model.save_pretrained(destination, max_shard_size="2GB")
        tokenizer.save_pretrained(destination)
        license_path = hf_hub_download(args.model, "LICENSE", revision=args.revision)
        shutil.copyfile(license_path, destination / "LICENSE")
        files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in destination.iterdir() if path.is_file()}
        (args.output / "base-provenance.json").write_text(json.dumps({
            "base_model": args.model, "revision": args.revision, "quantized_nf4": args.quantize,
            "files_sha256": files,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
