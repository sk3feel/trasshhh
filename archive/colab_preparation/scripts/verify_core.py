"""Extraction checks; historical imports are used only for this equivalence comparison."""

import gc
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/night_20261022"))

import core as old_core
import numpy as np
import torch
from tokenizers import Tokenizer
from engine import load_model as old_load_model

from src.data import LengthBuckets, TokenDataset, collate
from src.evaluate import evaluate_variant, validation_inputs, validation_loss
from src.generate import generate_questions, sha256_file
from src.model import configure_runtime, load_model, verify_model_state


def main():
    bundle = ROOT / "artifacts/baseline"
    report = bundle / "reproduction"
    report.mkdir(exist_ok=True)
    identity = json.loads((bundle / "identity_before.json").read_text())
    for name, expected in identity["files"].items():
        assert sha256_file(ROOT / name) == expected, name
    configure_runtime()
    old_core.ART = ROOT / "data/processed/original"
    old_model, checkpoint = old_load_model(bundle / "resume.pt")
    model = load_model(bundle)
    model_check = verify_model_state(model, checkpoint)
    settings = json.loads((ROOT / "configs/generation_greedy.json").read_text())
    questions, references = validation_inputs(bundle, old_core.ART)
    indices = np.load(bundle / "generation_512_indices.npy")[:32]
    old_predictions, _ = old_core.generate_rows(
        old_model, old_core.Tokens("validation", indices), batch=32, max_new_tokens=256
    )
    tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    new_predictions, _ = generate_questions(model, tokenizer, questions[:32], settings)
    keys = ["code", "tokens", "eos", "hit_limit"]
    assert all(
        all(old[key] == new[key] for key in keys)
        for old, new in zip(old_predictions, new_predictions)
    )
    old_dataset = old_core.Tokens("validation", indices)
    new_dataset = TokenDataset(old_core.ART, "validation", indices)
    old_batch = old_core.collate([old_dataset[i] for i in range(32)])
    new_batch = collate([new_dataset[i] for i in range(32)])
    assert all(torch.equal(value, new_batch[key]) for key, value in old_batch.items())
    train = TokenDataset(ROOT / "data/processed/training")
    old_buckets = old_core.Buckets(train, 64, 42, 4)
    new_buckets = LengthBuckets(train, 64, 42, 4)
    assert list(itertools.islice(old_buckets, 20)) == list(
        itertools.islice(new_buckets, 20)
    )
    del old_model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    result = evaluate_variant(
        model, tokenizer, questions, references, settings, report / "greedy"
    )
    metrics = result["metrics"]
    assert metrics["proxy_denominator"] == 493
    historical = json.loads(
        (
            ROOT
            / "experiments/night_20261022/evaluations/selected_generation/predictions_512_checkpoint.json"
        ).read_text()
    )
    assert all(
        a["code"] == b["code"]
        for a, b in zip(result["predictions"], historical["predictions"])
    )
    assert metrics["proxy_matches"] == 330, metrics
    ce = validation_loss(model, old_core.ART)
    assert abs(ce["validation_ce"] - 0.2955000123701405) < 1e-6, ce
    audit = {
        "model": model_check,
        "asset_hashes_unchanged": True,
        "old_new_greedy_subset_identical": 32,
        "full512_programs_identical_to_historical_baseline": True,
        "batch_masks_labels_decoder_shift_identical": True,
        "first20_epoch4_batches_identical": True,
        "validation": metrics,
        **ce,
        "GPU": torch.cuda.get_device_name(),
        "training_updates_performed": 0,
    }
    (report / "checks.json").write_text(json.dumps(audit, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--resume",
            str(bundle / "resume.pt"),
            "--check-resume",
            "--audit-output",
            str(report / "resume_check.json"),
        ],
        cwd=ROOT,
        check=True,
    )
    assert (
        sha256_file(bundle / "resume.pt")
        == identity["files"]["artifacts/baseline/resume.pt"]
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
