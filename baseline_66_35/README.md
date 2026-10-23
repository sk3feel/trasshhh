# Исходный baseline greedy: public 637/960 со слов пользователя

Здесь сохранён прежний BART **20 106 240 параметров**, greedy **330/493** на локальной fixed512 validation, CE **0.2955000124**. Его public **637/960 = 66,35%** и T4 generation27.31s / process79.46s сообщены пользователем. **Исходный файл именно scored-посылки и её сырые T4-логи не найдены**, поэтому независимо подтвердить привязку public к байтам посылки невозможно. Проверена идентичность весов прежнему frozen baseline по manifests, SHA и существующему tensor audit. Модель не подменялась.

Это не новый overnight checkpoint333/493 и не beam-вариант public644/960. Beam использовал те же веса, но другое декодирование; здесь рабочий inference notebook фиксирует greedy.

## Происхождение и неизменяемые артефакты

- `resume.pt`: полный исходный checkpoint, SHA256 `a677bc9d0834907466e20728ef08712ab919c12b61292d57cc5ffe38a5f482c2`.
- `model/model.safetensors`: SHA256 `173f50c8468d145599353c88170d4cb7e203f3aefea8f11638d7f8140cf8af35`.
- `tokenizer.json`: frozen train-only BPE8192, SHA256 `4b25bf57faf2b1ce8b143850c12993864cdff7e8ec0d565f1dd7dad466a4d2b0`.
- Исторический source: `../archive/experiments/night_20261022/selected/best_generation.pt`, конец late_continuation. До выделения модели было11 819 810 предъявлений, не уникальных задач.
- `bundle_manifest.json`, `identity_before.json`, `reproduction/` — исходные, не переписанные свидетельства. Их старые пути переведены в [PATH_MAP](../archive/PATH_MAP.json).
- `provenance.json` отделяет подтверждённую идентичность весов от сообщённого public score.

Архитектура: encoder4/decoder4,d384,heads6,FFN1536, learned positions512, dropout0.1, shared encoder/decoder embedding и tied LM head. Случайная исходная инициализация, внешние pretrained weights отсутствуют. PAD/BOS/EOS/UNK0/1/2/3. Source `[BOS]+question+[EOS]`≤192, target `code+[EOS]`≤384. Greedy deployment: batch64, FP16, SDPA, KV-cache, max_new_tokens256, forced EOS отсутствует. Локальный330/493 относится к batch32; численные различия между batch/hardware известны.

## Код и команды

`src/` — минимальное прежнее ядро: `prepare.py` (исходная выборка, filters, split, BPE), `data.py` (mmap/батчи), `model.py` (random BART/own weights), `train.py` (прежний late-resume loop), `generate.py`, `evaluate.py`, `sandbox.py`, `submission.py`. Код не импортирует модули из archive. Исторические рецепты ранних этапов и неизменные отчёты остаются в archive; этот loop не является реализацией нового106M Colab-эксперимента.

Все команды ниже выполняются **из baseline_66_35/**, ничего автоматически не запускается. В CLI всегда переданы явные новые пути: старые default paths оставлены в исходниках как исторические. Воспроизведение окружения — requirements-train/prepare/inference; существующие venv находятся уровнем выше.

```bash
cd baseline_66_35
# Только inference, новые outputs вне frozen baseline:
../.venv-experiments/bin/python -m src.generate --bundle . \
  --generation-config configs/generation_greedy_t4.json \
  --output-dir ../data/outputs/baseline_greedy_new

# При необходимости numerical validation; generated code только в Docker:
../.venv-experiments/bin/python -m src.evaluate --bundle . \
  --validation ../data/processed/original \
  --generation-config configs/generation_greedy.json \
  --output ../data/outputs/baseline_validation_new --ce

# Проверка восстановления без optimizer update:
../.venv-experiments/bin/python -m src.train \
  --config configs/baseline.local.json --resume resume.pt --check-resume

# Подготовка только по будущему отдельному решению, в НОВЫЙ каталог:
../.venv-analysis/bin/python -m src.prepare initial \
  --raw-train ../data/tinysms_qa_12m/data/train --output ../data/new_initial
```

`configs/baseline.json` оставлен неизменным историческим config. `configs/baseline.local.json` отличается только путями после переноса. `generation_greedy.json` — локальный batch32; `generation_greedy_t4.json` — deployment batch64. Beam configs сохранены для provenance, не выбраны default notebook.

## Kaggle inference

`inference_bundle/` содержит только greedy own weights, tokenizer, question-only1919, два greedy config, inference requirements и нужные локальные модули. Большие одинаковые файлы — hard links, не дублируют дисковое содержимое. Загрузите **содержимое inference_bundle** как Private Kaggle Dataset, импортируйте `Kaggle_inference.ipynb`, подключите Dataset, выберите одну T4 и Run all. Notebook обнаруживает bundle по manifest, использует greedy, сохраняет `submission.jsonl` и `kaggle_results.zip`. Он не зависит от archive. Сам новый облачный запуск после переноса не выполнен.

`evidence/local_greedy_rerun/` содержит позднее локальное воспроизведение1919 и timing на RTX3060. **Это не оригинальная public-посылка и не T4-замер.** Не выдавайте эти файлы за исходные бот-артефакты. Frozen weights/checkpoint не использовать как output directory и не перезаписывать.
