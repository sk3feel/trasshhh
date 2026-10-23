"""Evaluate exported validation generations locally, only inside existing Docker sandbox."""

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sandbox import compare, execute


def evaluate(archive, output, workers=4):
    expected = json.loads((ROOT / "configs/colab_data_identity.json").read_text())
    with zipfile.ZipFile(archive) as z:
        # Read specific members, never extract or import code from the review ZIP.
        for name, digest in expected["evaluation_files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("Frozen validation checksum mismatch: " + name)
        questions = json.loads(z.read("evaluation/questions_512.json"))
        references = json.loads(z.read("evaluation/references_512.json"))
        if len(questions) != 512 or len(references) != 512:
            raise ValueError("Expected fixed 512 questions/references")
        if sum(r["status"] == "ok" and math.isfinite(r["value"]) for r in references) != 493:
            raise ValueError("Expected 493 finite references")
        meta = json.loads(z.read("review_metadata.json"))
        exports = meta["status"].get("verified_exports", [])
        output.mkdir(parents=True, exist_ok=True)
        results, skipped = [], []
        for export in exports:
            name = export["name"]
            if Path(name).name != name:
                raise ValueError("Invalid export name")
            member = "validation_predictions/" + name + "_512.json"
            if member not in z.namelist():
                skipped.append({"name": name, "reason": "512 generation absent; not scored"})
                continue
            record = json.loads(z.read(member))
            if record["identity"] != meta["identity"]:
                raise ValueError("Generation identity mismatch")
            rows = record["predictions"]
            if len(rows) != 512:
                raise ValueError("Wrong prediction count")
            for i, (row, question) in enumerate(zip(rows, questions)):
                if (row["index"] != i or not isinstance(row["code"], str)
                        or row["question_sha256"] != hashlib.sha256(question.encode()).hexdigest()):
                    raise ValueError("Question order/content mismatch")
            execution = execute([r["code"] for r in rows], workers=workers)
            metrics = {"export": name, "validation_ce": export["CE"],
                       **compare(execution, references), "official_pass_at_1": None}
            (output / (name + "_execution.json")).write_text(json.dumps(execution, indent=2))
            (output / (name + "_metrics.json")).write_text(json.dumps(metrics, indent=2))
            results.append(metrics)
            print(json.dumps(metrics), flush=True)
        results.sort(key=lambda r: (-r["proxy_matches"], r["validation_ce"] if r["validation_ce"] is not None else float("inf")))
        summary = {"ranked_exports": results, "skipped": skipped,
                   "selection": "local noisy numerical reference agreement; not official pass@1",
                   "public_pass_at_1": None, "Drive_exports_automatically_modified": False}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        if not results:
            raise RuntimeError("No verified export with fixed512 predictions; see summary.json")
        return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    evaluate(args.review, args.output, args.workers)
