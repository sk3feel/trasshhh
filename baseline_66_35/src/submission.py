"""Validate unchanged model outputs and write the bot's one-field JSONL format."""

from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import zipfile


def export_submission(bundle, output, official_questions=None):
    bundle, output = Path(bundle), Path(output)
    inputs = [
        json.loads(line)
        for line in (bundle / "test_questions.jsonl").read_text().splitlines()
    ]
    records = json.loads((output / "predictions.json").read_text())
    if len(inputs) != 1919 or len(records) != 1919:
        raise ValueError("Exactly 1919 questions and predictions are required")
    for index, (row, record) in enumerate(zip(inputs, records)):
        expected = hashlib.sha256(row["question"].encode()).hexdigest()
        if row["index"] != index or record["index"] != index:
            raise ValueError("Prediction order differs from original input")
        if record["question_sha256"] != expected or not isinstance(record["code"], str):
            raise ValueError("Question hash or code type mismatch")
    order = list(range(1919))
    if official_questions is not None:
        official_questions = Path(official_questions)
        rows = [
            json.loads(line) for line in official_questions.read_text().splitlines()
        ]
        if len(rows) != 1919:
            raise ValueError("Official question count differs")
        positions = defaultdict(deque)
        for index, row in enumerate(inputs):
            positions[row["question"]].append(index)
        order = []
        for row in rows:
            text = row if isinstance(row, str) else row.get("question")
            if not isinstance(text, str) or not positions[text]:
                raise ValueError(
                    "Official question texts differ; do not submit these outputs"
                )
            order.append(positions[text].popleft())
        if any(positions.values()):
            raise ValueError("Official question multiplicities differ")
    destination = output / "submission.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for index in order:
            stream.write(
                json.dumps({"code": records[index]["code"]}, ensure_ascii=False) + "\n"
            )
    parsed = [json.loads(line) for line in temporary.read_text().splitlines()]
    assert len(parsed) == 1919
    assert all(
        set(row) == {"code"} and row["code"] == records[index]["code"]
        for row, index in zip(parsed, order)
    )
    temporary.replace(destination)
    validation = {
        "lines": 1919,
        "exact_keys": ["code"],
        "programs_rewritten_or_truncated": False,
        "input_order_verified": True,
        "official_pass1_file_compared": official_questions is not None,
        "reordered_to_official_order": order != list(range(1919)),
        "submission_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "official_questions_sha256": (
            hashlib.sha256(official_questions.read_bytes()).hexdigest()
            if official_questions
            else None
        ),
        "submission_sent": False,
    }
    (output / "submission_validation.json").write_text(json.dumps(validation, indent=2))
    with zipfile.ZipFile(
        output / "kaggle_results.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in [
            "submission.jsonl",
            "predictions.json",
            "submission_validation.json",
            "timing.json",
            "generation_parameters.json",
            "input_alignment.json",
            "inference.log",
        ]:
            if (output / name).exists():
                archive.write(output / name, arcname=name)
    return validation
