"""Fixed-denominator execution-result reference agreement, never textual exact match."""

import argparse
import ast
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from .data import TokenDataset, collate
from .generate import generate_questions
from .model import configure_runtime, load_model
from .sandbox import compare, execute


def validation_loss(model, directory):
    dataset = TokenDataset(directory, "validation")
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate)
    total, count = 0.0, 0
    model.eval()
    for batch in loader:
        batch.pop("_indices")
        batch = {name: value.cuda() for name, value in batch.items()}
        tokens = int(batch["labels"].ne(-100).sum())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batch, use_cache=False).loss
        total += float(loss) * tokens
        count += tokens
    return {"validation_ce": total / count, "validation_target_tokens": count}


def syntax_ok(code):
    try:
        ast.parse(code)
        return bool(code.strip())
    except (SyntaxError, ValueError, RecursionError):
        return False


def evaluate_variant(
    model, tokenizer, questions, references, settings, output, greedy=None
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "generation_parameters.json").write_text(json.dumps(settings, indent=2))
    predictions, metrics = generate_questions(model, tokenizer, questions, settings)
    execution = execute([record["code"] for record in predictions])
    metrics.update(compare(execution, references))
    correct = [
        ref["status"] == result["status"] == "ok"
        and math.isclose(result["value"], ref["value"], rel_tol=1e-6, abs_tol=1e-6)
        for result, ref in zip(execution, references)
    ]
    syntax = [syntax_ok(record["code"]) for record in predictions]
    metrics.update(
        {
            "metric_name": "finite_scalar_execution_reference_agreement",
            "syntax_errors": sum(not value for value in syntax),
            "runtime_errors_excluding_syntax": sum(
                valid
                and result["status"]
                in ["runtime_error", "process_error", "timeout", "missing_callable"]
                for valid, result in zip(syntax, execution)
            ),
            "generated_status_counts": dict(
                Counter(result["status"] for result in execution)
            ),
            "unique_programs": len({record["code"] for record in predictions}),
            "relative_to_greedy_fixed": (
                sum(a and not b for a, b in zip(correct, greedy))
                if greedy is not None
                else 0
            ),
            "relative_to_greedy_broken": (
                sum(b and not a for a, b in zip(correct, greedy))
                if greedy is not None
                else 0
            ),
            "official_pass_at_1": None,
        }
    )
    artifact = {
        "predictions": predictions,
        "execution": execution,
        "correct": correct,
        "metrics": metrics,
    }
    (output / "results.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2)
    )
    return artifact


def validation_inputs(bundle, directory):
    indices = np.load(Path(bundle) / "generation_512_indices.npy")
    # Only question text enters generation; code/reference values are evaluation-only.
    questions = pq.read_table(
        Path(directory) / "validation.parquet", columns=["question"]
    )["question"]
    texts = [questions[int(index)].as_py() for index in indices]
    references = json.loads((Path(bundle) / "references_512.json").read_text())
    assert len(texts) == len(references) == 512
    assert sum(row["status"] == "ok" for row in references) == 493
    return texts, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/processed/original")
    )
    parser.add_argument(
        "--generation-config", type=Path, default=Path("configs/generation_greedy.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ce", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    model = load_model(args.bundle)
    tokenizer = Tokenizer.from_file(str(args.bundle / "tokenizer.json"))
    questions, references = validation_inputs(args.bundle, args.validation)
    settings = json.loads(args.generation_config.read_text())
    result = evaluate_variant(
        model, tokenizer, questions, references, settings, args.output
    )
    if args.ce:
        (args.output / "validation_ce.json").write_text(
            json.dumps(validation_loss(model, args.validation), indent=2)
        )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
