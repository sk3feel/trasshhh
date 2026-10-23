"""Question-only generation shared by local evaluation and Kaggle inference."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import torch
from tokenizers import Tokenizer

from .model import configure_runtime, load_model


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def read_questions(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(rows) != 1919:
        raise ValueError("Expected the unchanged 1919 test questions")
    for index, row in enumerate(rows):
        if set(row) != {"index", "question"} or row["index"] != index:
            raise ValueError("Unexpected question schema or order")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("Empty or non-string question")
    return rows


def generate_questions(model, tokenizer, questions, settings):
    """Preserve old BOS/EOS/PAD, padding, FP16, cache and greedy computations."""
    batch_size = settings["batch_size"]
    if settings["do_sample"] or settings["max_new_tokens"] != 256:
        raise ValueError("This comparison fixes deterministic decoding and length 256")
    arguments = {
        "do_sample": False,
        "num_beams": settings["num_beams"],
        "use_cache": True,
        "max_new_tokens": 256,
        "forced_eos_token_id": None,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "decoder_start_token_id": 1,
    }
    if settings["num_beams"] > 1:
        arguments["length_penalty"] = settings["length_penalty"]
        arguments["early_stopping"] = settings["early_stopping"]
    records = []
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(questions), batch_size):
        texts = questions[offset : offset + batch_size]
        encoded = [[1] + row.ids + [2] for row in tokenizer.encode_batch(texts)]
        if max(map(len, encoded)) > 192:
            raise ValueError("Input exceeds source limit 192; no silent truncation")
        padded = ((max(map(len, encoded)) + 7) // 8) * 8
        ids = torch.zeros((len(encoded), padded), dtype=torch.long, device="cuda")
        for index, row in enumerate(encoded):
            ids[index, : len(row)] = torch.tensor(row, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs = model.generate(
                input_ids=ids, attention_mask=ids.ne(0), **arguments
            )
        for index, row in enumerate(outputs.cpu().tolist()):
            row = row[1:]
            ended = 2 in row
            if ended:
                row = row[: row.index(2) + 1]
            records.append(
                {
                    "index": offset + index,
                    "code": tokenizer.decode(row, skip_special_tokens=True),
                    "tokens": len(row),
                    "eos": ended,
                    "hit_limit": not ended and len(row) >= 256,
                }
            )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    lengths = [record["tokens"] for record in records]
    return records, {
        "generation_seconds": seconds,
        "generation_examples": len(records),
        "mean_output_tokens": sum(lengths) / len(lengths),
        "p95_output_tokens": percentile(lengths, 0.95),
        "hit_limit": sum(record["hit_limit"] for record in records),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
        "settings": settings,
        "timing_includes": "tokenization, batching, transfers, GPU generation, decoding",
    }


def run(bundle, output, settings):
    started = time.perf_counter()
    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    for name, expected in manifest["asset_sha256"].items():
        if sha256_file(bundle / name) != expected:
            raise ValueError(f"Frozen asset checksum mismatch: {name}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    configure_runtime()
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "transformers", "tokenizers", "numpy", "safetensors"]
    }
    rows = read_questions(bundle / "test_questions.jsonl")
    print(json.dumps({"versions": versions, "gpu": torch.cuda.get_device_name(0)}))
    torch.cuda.synchronize()
    loading = time.perf_counter()
    model = load_model(bundle)
    tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    assert [tokenizer.token_to_id(t) for t in ["<pad>", "<bos>", "<eos>", "<unk>"]] == [
        0,
        1,
        2,
        3,
    ]
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - loading
    questions = [row["question"] for row in rows]
    _, warm = generate_questions(
        model, tokenizer, questions[:8], {**settings, "batch_size": 8}
    )
    records, timing = generate_questions(model, tokenizer, questions, settings)
    assert len(records) == 1919
    for record, row in zip(records, rows):
        assert record["index"] == row["index"]
        record["question_sha256"] = hashlib.sha256(row["question"].encode()).hexdigest()
    (output / "predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )
    with (output / "submission.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps({"code": record["code"]}, ensure_ascii=False) + "\n"
            )
    assert len((output / "submission.jsonl").read_text().splitlines()) == 1919
    alignment = {
        "expected_count": 1919,
        "actual_count": len(records),
        "indices_in_original_order": True,
        "input_sha256": sha256_file(bundle / "test_questions.jsonl"),
        "official_pass1_file_compared": False,
    }
    (output / "input_alignment.json").write_text(json.dumps(alignment, indent=2))
    timing.update(
        {
            "load_model_and_tokenizer_seconds": load_seconds,
            "warmup_seconds": warm["generation_seconds"],
            "warmup_examples": 8,
            "pipeline_seconds": time.perf_counter() - started,
            "pipeline_excludes": "interpreter/import startup and dependency installation",
            "versions": versions,
            "one_T4_measured": "T4" in timing["gpu"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "submission_sent": False,
        }
    )
    (output / "timing.json").write_text(json.dumps(timing, indent=2))
    (output / "generation_parameters.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps(timing, indent=2))
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/generation_selected.json"),
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    run(args.bundle, args.output_dir, json.loads(args.generation_config.read_text()))


if __name__ == "__main__":
    main()
