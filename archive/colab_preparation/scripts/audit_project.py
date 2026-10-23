"""Final structural/input/output checks. No training or generated-code execution."""

import ast
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    import sys

    sys.path.insert(0, str(ROOT))
    from src.submission import export_submission
    from src.sandbox import execute

    baseline = ROOT / "artifacts/baseline"
    identity = json.loads((baseline / "identity_before.json").read_text())
    for path, expected in identity["files"].items():
        assert sha(ROOT / path) == expected, path
    for path in (ROOT / "src").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("archive", "experiments"))
            if isinstance(node, ast.Import):
                assert all(
                    not n.name.startswith(("archive", "experiments"))
                    for n in node.names
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ["exec", "eval"], path
    # Historical probe strings may contain exec, but no host AST call does.
    original = ast.parse(
        (ROOT / "archive/experiments/session_20261021/prepare.py").read_text()
    )
    current = ast.parse((ROOT / "src/prepare.py").read_text())
    for name in ["group", "is_validation", "technical_reason", "stats"]:
        a = next(
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        b = next(
            n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert ast.dump(a, include_attributes=False) == ast.dump(
            b, include_attributes=False
        ), name
    for path in (ROOT / "experiments").iterdir():
        if path.name == "decoding":
            continue
        assert path.is_symlink() and path.exists(), path
    old_sandbox = ast.parse(
        (ROOT / "archive/experiments/night_20261022/sandbox.py").read_text()
    )
    new_sandbox = ast.parse((ROOT / "src/sandbox.py").read_text())
    for name in ["CHILD", "PARENT"]:
        old = next(
            n
            for n in old_sandbox.body
            if isinstance(n, ast.Assign) and n.targets[0].id == name
        )
        new = next(
            n
            for n in new_sandbox.body
            if isinstance(n, ast.Assign) and n.targets[0].id == name
        )
        assert ast.dump(
            ast.parse(ast.literal_eval(old.value)), include_attributes=False
        ) == ast.dump(ast.parse(ast.literal_eval(new.value)), include_attributes=False)
    probes = execute(["def solve():\n    return -0.5", "def solve():\n    return 1/0"])
    assert probes[0] == {"status": "ok", "value": -0.5}
    assert probes[1]["status"] == "runtime_error"
    questions = [
        json.loads(line)
        for line in (ROOT / "kaggle_bundle/test_questions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(questions) == 1919 and all(
        row["index"] == i and set(row) == {"index", "question"}
        for i, row in enumerate(questions)
    )
    greedy = json.loads(
        (ROOT / "artifacts/deployment_test/greedy/predictions.json").read_text()
    )
    historical = json.loads(
        (
            ROOT / "archive/experiments/night_20261022/inference/predictions.json"
        ).read_text()
    )
    assert len(greedy) == len(historical) == 1919
    keys = ["index", "code", "tokens", "eos", "hit_limit"]
    changed = [
        i
        for i, (a, b) in enumerate(zip(greedy, historical))
        if any(a[k] != b[k] for k in keys)
    ]
    assert not changed, changed
    validation = export_submission(
        ROOT / "kaggle_bundle", ROOT / "artifacts/deployment_test/selected"
    )
    # Verify only output serialization/order; no programs are executed.
    predictions = json.loads(
        (ROOT / "artifacts/deployment_test/selected/predictions.json").read_text()
    )
    with tempfile.TemporaryDirectory(prefix="hw2_submission_") as directory:
        temporary = Path(directory)
        (temporary / "predictions.json").write_text(json.dumps(predictions))
        official = temporary / "official.jsonl"
        official.write_text(
            "".join(
                json.dumps({"question": row["question"]}) + "\n"
                for row in questions[::-1]
            )
        )
        checked = export_submission(ROOT / "kaggle_bundle", temporary, official)
        rows = [
            json.loads(line)
            for line in (temporary / "submission.jsonl").read_text().splitlines()
        ]
        assert checked["reordered_to_official_order"] and all(
            a["code"] == b["code"] for a, b in zip(rows, predictions[::-1])
        )
        official.write_text(
            json.dumps({"question": "mismatched question"})
            + "\n"
            + "".join(
                json.dumps({"question": row["question"]}) + "\n"
                for row in questions[1:]
            )
        )
        try:
            export_submission(ROOT / "kaggle_bundle", temporary, official)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched official questions must fail")
    notebook = json.loads((ROOT / "notebooks/T4_inference.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")
    result = {
        "baseline_and_input_hashes_unchanged": True,
        "current_src_imports_archive": False,
        "host_exec_or_eval_calls_in_src": False,
        "sandbox_payload_asts_preserved": True,
        "sandbox_numeric_error_smoke_passed": True,
        "historical_filter_group_helpers_ast_identical": True,
        "history_compatibility_paths_valid": True,
        "new_greedy64_identical_to_historical_full1919": True,
        "selected_submission_validation": validation,
        "official_order_reversal_test_passed": True,
        "official_question_mismatch_rejected": True,
        "notebook_code_cells_compile": True,
        "notebook_remote_Kaggle_T4_executed": False,
        "source_module_lines": {
            p.name: len(p.read_text().splitlines())
            for p in sorted((ROOT / "src").glob("*.py"))
        },
        "selected_local_3060_timing": json.loads(
            (ROOT / "artifacts/deployment_test/selected/timing.json").read_text()
        ),
        "greedy_local_3060_timing": json.loads(
            (ROOT / "artifacts/deployment_test/greedy/timing.json").read_text()
        ),
        "new_training_updates": 0,
        "new_bot_submissions": 0,
    }
    (ROOT / "artifacts/project_audit.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if not k.endswith("_timing")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
