"""Build small current-code review and local-only Kaggle bundles; never trains."""

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def copy(path, destination, hardlink=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if hardlink:
        os.link(path, destination)
    else:
        shutil.copy2(path, destination)


def notebook():
    cells = []

    def md(text):
        cells.append(
            {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}
        )

    def code(text):
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "outputs": [],
                "metadata": {},
                "source": text.splitlines(True),
            }
        )

    md(
        """# NLP HW2 — выбранное декодирование и greedy на одной T4

Загрузите kaggle_bundle.zip как Private Kaggle Dataset, подключите к notebook и включите GPU T4. Запустите с чистого kernel. Только первая GPU используется даже при выдаче двух T4. MODE="selected" выбирает локального победителя; MODE="greedy" возвращает зафиксированный baseline. Официальный public637/960 и T4 27.31/79.46s относятся к **прежнему greedy-запуску**, сообщённому пользователем, не к новому выбранному режиму.

Notebook не обучает, не исполняет generated-программы и не отправляет посылки. 64 токена — рекомендация бота; сохранён проверенный предел256. Для бота скачивается submission.jsonl; остальные файлы — для проверки.
"""
    )
    code(
        """import os, sys, time
NOTEBOOK_STARTED = time.perf_counter()
if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
    raise RuntimeError('Restart the kernel; run this cell before initializing CUDA')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
"""
    )
    code(
        """from pathlib import Path
import json, subprocess, importlib.metadata
BUNDLE = None  # Optional explicit Path('/kaggle/input/your-dataset/kaggle_bundle')
MODE = 'selected'  # Change only this to 'greedy' for the fixed baseline
OFFICIAL_QUESTIONS = None  # Optional Path to pass1_questions.jsonl
OUTPUT = Path('/kaggle/working')
if BUNDLE is None:
    found, visited = [], set()
    for directory, subdirs, files in os.walk('/kaggle/input', followlinks=True):
        real = os.path.realpath(directory)
        if real in visited:
            subdirs[:] = []
            continue
        visited.add(real)
        if 'bundle_manifest.json' in files:
            path = Path(directory) / 'bundle_manifest.json'
            if json.loads(path.read_text()).get('bundle_kind') == 'hw2_kaggle_inference':
                found.append(path.parent.resolve())
        if OFFICIAL_QUESTIONS is None and 'pass1_questions.jsonl' in files:
            OFFICIAL_QUESTIONS = Path(directory) / 'pass1_questions.jsonl'
    if len(found) != 1:
        raise ValueError(f'Set BUNDLE explicitly; candidates: {found}')
    BUNDLE = found[0]
assert MODE in ['selected', 'greedy']
CONFIG_NAME = 'generation_selected.json' if MODE == 'selected' else 'generation_greedy_t4.json'
GENERATION_CONFIG = BUNDLE / 'configs' / CONFIG_NAME
print('Bundle:', BUNDLE, 'mode:', MODE)
print(GENERATION_CONFIG.read_text())
"""
    )
    code(
        """install_started = time.perf_counter()
required = {'transformers':'4.46.3', 'tokenizers':'0.20.3'}
repair = False
for name, version in required.items():
    try:
        repair |= importlib.metadata.version(name) != version
    except importlib.metadata.PackageNotFoundError:
        repair = True
for name in ['numpy', 'safetensors']:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        repair = True
if repair:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(BUNDLE/'requirements-inference.txt')], check=True)
DEPENDENCY_INSTALL_SECONDS = time.perf_counter() - install_started
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print('GPU:', torch.cuda.get_device_name(0), 'CUDA:', torch.version.cuda)
print('Python:', sys.version)
for name in ['torch','transformers','tokenizers','numpy','safetensors']:
    print(name, importlib.metadata.version(name))
if 'T4' not in torch.cuda.get_device_name(0):
    print('Not a T4: do not claim the official T4 time bound from this run.')
"""
    )
    md(
        """Полный дочерний процесс измеряется отдельно: Python/imports, проверка файлов, загрузка, warmup, генерация и запись. В timing.json также раздельно записаны load, warmup и warmed generation; warmup не скрывается. Установка зависимостей измеряется отдельно. Для выбранного beam режима окончательный замер на T4 ещё необходим.
"""
    )
    code(
        """OUTPUT.mkdir(exist_ok=True, parents=True)
started = time.perf_counter()
command = [sys.executable, str(BUNDLE/'generate.py'), '--bundle', str(BUNDLE),
           '--generation-config', str(GENERATION_CONFIG), '--output-dir', str(OUTPUT)]
with (OUTPUT/'inference.log').open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
elapsed = time.perf_counter() - started
print((OUTPUT/'inference.log').read_text())
result.check_returncode()
timing = json.loads((OUTPUT/'timing.json').read_text())
timing['subprocess_end_to_end_seconds'] = elapsed
timing['dependency_install_seconds'] = DEPENDENCY_INSTALL_SECONDS
timing['mode'] = MODE
timing['one_T4_confirmed'] = 'T4' in torch.cuda.get_device_name(0)
timing['full_process_under_180_seconds'] = elapsed <= 180
timing['notebook_elapsed_including_setup_seconds'] = time.perf_counter() - NOTEBOOK_STARTED
(OUTPUT/'timing.json').write_text(json.dumps(timing, indent=2))
print(json.dumps(timing, indent=2))
"""
    )
    code(
        """# Validate 1919 unchanged outputs; optional matching to official question order.
sys.path.insert(0, str(BUNDLE))
from src.submission import export_submission
validation = export_submission(BUNDLE, OUTPUT, OFFICIAL_QUESTIONS)
print(json.dumps(validation, indent=2))
if not validation['official_pass1_file_compared']:
    print('Official pass1_questions.jsonl is absent: exact order against that separate file is not verified. Bundled original test order is preserved.')
from IPython.display import display, FileLink
os.chdir(OUTPUT)
display(FileLink('submission.jsonl'))
display(FileLink('kaggle_results.zip'))
print('Download submission.jsonl for the bot. Return kaggle_results.zip and the executed notebook for review.')
"""
    )
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "T4_inference.ipynb", "exec")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build():
    baseline = ROOT / "artifacts/baseline"
    kaggle = ROOT / "kaggle_bundle"
    review = ROOT / "review_bundle"
    kaggle.mkdir(exist_ok=True)
    review.mkdir(exist_ok=True)
    for path in (baseline / "model").iterdir():
        copy(path, kaggle / "model" / path.name, hardlink=True)
    for name in ["tokenizer.json", "test_questions.jsonl"]:
        copy(baseline / name, kaggle / name, hardlink=True)
    for name in ["__init__.py", "model.py", "generate.py", "submission.py"]:
        copy(ROOT / "src" / name, kaggle / "src" / name)
    for name in [
        "generation_greedy.json",
        "generation_greedy_t4.json",
        "generation_selected.json",
        "generation_hf_defaults.json",
    ]:
        copy(ROOT / "configs" / name, kaggle / "configs" / name)
    copy(ROOT / "requirements-inference.txt", kaggle / "requirements-inference.txt")
    (kaggle / "generate.py").write_text(
        '''"""Single-GPU local-only entry point; no training or downloads of model weights."""\nimport os\nos.environ['CUDA_VISIBLE_DEVICES']='0'\nos.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\nfrom src.generate import main\nif __name__=='__main__':\n    main()\n'''
    )
    manifest = json.loads((baseline / "bundle_manifest.json").read_text())
    manifest["asset_sha256"] = {
        name: sha(kaggle / name)
        for name in [
            "model/config.json",
            "model/generation_config.json",
            "model/model.safetensors",
            "tokenizer.json",
            "test_questions.jsonl",
            "configs/generation_greedy.json",
            "configs/generation_greedy_t4.json",
            "configs/generation_hf_defaults.json",
            "configs/generation_selected.json",
        ]
    }
    manifest["generation_historical_greedy"] = manifest.pop("generation", {})
    manifest["generation_selected"] = json.loads(
        (ROOT / "configs/generation_selected.json").read_text()
    )
    manifest["baseline_results"] = json.loads(
        (baseline / "identity_before.json").read_text()
    )["user_reported_external_results"]
    manifest["selected_decoding"] = json.loads(
        (ROOT / "experiments/decoding/selection.json").read_text()
    )
    manifest["source_checkpoint"] = "artifacts/baseline/resume.pt"
    manifest["checkpoint_metrics"]["official_pass_at_1"] = 637 / 960
    manifest["checkpoint_metrics"][
        "official_pass_source"
    ] = "User-reported prior greedy submission637/960; not a selected-beam measurement"
    write_json(kaggle / "bundle_manifest.json", manifest)
    nb = notebook()
    write_json(ROOT / "notebooks/T4_inference.ipynb", nb)
    write_json(kaggle / "T4_inference.ipynb", nb)
    (kaggle / "README.md").write_text(
        """# Kaggle inference only

1. Upload kaggle_bundle.zip (or extracted contents) as a **Private Kaggle Dataset**. Preserve model/, tokenizer.json, src/, configs/ and test_questions.jsonl. No train data or optimizer state is required.
2. Import T4_inference.ipynb as a Notebook, attach that Dataset and choose GPU T4. Start with a clean kernel. The first cell restricts CUDA visibility to one GPU before initialization, even if Kaggle supplies two T4s.
3. MODE="selected" selects the local-validation winner; change only MODE="greedy" to reproduce the fixed baseline. Exact parameters are configs/generation_selected.json and generation_greedy_t4.json. The separate validation comparison uses batch32; prior greedy deployment uses batch64. Any deployment batch adjustment is documented in experiments/decoding in the review archive.
4. Run All. Versions/GPU are printed. Pip may need Internet; CUDA-enabled torch is provided by Kaggle and not replaced automatically. Our own model weights are loaded with local_files_only=True. No Comet/API keys, training data, generated-code execution or automatic submissions.
5. Download submission.jsonl for the bot:1919 JSONL rows, exactly one code field per row. Return kaggle_results.zip and executed notebook for review. The ZIP contains predictions.json, timing.json, generation_parameters.json, input_alignment.json, submission_validation.json and inference.log in addition to the submission.

The test file is UTF-8 JSONL {index,question},1919 original unmodified question strings in original parquet order. The index is derived0-based position, not an official task ID. Gold/reference answers are absent. Per-question hashes verify alignment. Optional attached pass1_questions.jsonl is checked by exact question text and used for explicit submission order; generated programs are never edited/truncated. Without that separate file its order is not directly verified.

Reported earlier greedy baseline: T4 generation27.31s, whole process79.46s, public637/960. These are user-provided prior measurements, not new selected-beam measurements. Selected mode must still be timed on one T4. timing.json separates load, warmup, warmed generation, pipeline and full subprocess; dependencies are measured separately by the notebook.64 tokens is only the bot recommendation; the verified max_new_tokens=256 is retained.

Third-party packages/torch runtime are not bundled. Kaggle library/CUDA compatibility, the selected-mode180s bound and its official pass@1 remain unverified until your run. Floating-point outputs can differ across hardware/library versions. No claim of official improvement is made from local validation.
"""
    )
    # The review delivery includes current code only; all history remains on disk in archive/.
    for path in (ROOT / "src").glob("*.py"):
        copy(path, review / "src" / path.name)
    for path in (ROOT / "configs").glob("*.json"):
        copy(path, review / "configs" / path.name)
    for path in [
        ROOT / "README.md",
        ROOT / "requirements-train.txt",
        ROOT / "requirements-inference.txt",
        ROOT / "requirements-prepare.txt",
        ROOT / "archive/PATH_MAP.json",
        ROOT / "AGENT_USAGE.md",
    ]:
        copy(
            path,
            review
            / ("archive/PATH_MAP.json" if path.name == "PATH_MAP.json" else path.name),
        )
    copy(ROOT / "notebooks/T4_inference.ipynb", review / "notebooks/T4_inference.ipynb")
    for path in [
        baseline / "identity_before.json",
        baseline / "reproduction/checks.json",
        baseline / "reproduction/resume_check.json",
    ]:
        copy(path, review / "verification" / path.name)
    for name in [
        "results.csv",
        "results.json",
        "selection.json",
        "session.json",
        "REPORT.md",
        "deployment_batch_check.json",
    ]:
        copy(
            ROOT / "experiments/decoding" / name, review / "experiments/decoding" / name
        )
    for name in [
        "download_dataset.py",
        "verify_dataset.py",
    ]:
        copy(ROOT / "scripts" / name, review / "scripts" / name)
    copy(
        ROOT / "experiments/decoding/compare.py",
        review / "experiments/decoding/compare.py",
    )
    copy(
        ROOT / "experiments/decoding/deployment_batch.py",
        review / "experiments/decoding/deployment_batch.py",
    )

    for path in (ROOT / "experiments/decoding").glob(
        "*/effective_generation_parameters.json"
    ):
        copy(path, review / path.relative_to(ROOT))
    copy(
        ROOT / "artifacts/project_audit.json",
        review / "verification/project_audit.json",
    )
    copy(
        ROOT / "artifacts/baseline/reproduction/resume_check_final.json",
        review / "verification/resume_check_final.json",
    )


def finalize():
    results = []
    for name in ["review_bundle", "kaggle_bundle"]:
        folder = ROOT / name
        for cache in folder.rglob("__pycache__"):
            shutil.rmtree(cache)
        checksums = {
            str(p.relative_to(folder)): sha(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "FILES_SHA256.json"
        }
        write_json(folder / "FILES_SHA256.json", checksums)
        path = ROOT / f"{name}.zip"
        with zipfile.ZipFile(
            path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    archive.write(file, arcname=file.relative_to(ROOT))
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        results.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        )
    write_json(ROOT / "artifacts/delivery_archives.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    build()
    finalize()
