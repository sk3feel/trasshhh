# NLP HW2: текущий pipeline, frozen baseline и decoding

Актуальный код — **src/**. Старые результаты физически находятся в archive/experiments/; experiments/session_20261021, night_20261022 и day_20261022 — совместимые ссылки. В src нет импортов исторического кода. Карта переноса — [archive/PATH_MAP.json](archive/PATH_MAP.json). Исторические отчёты не переписаны; их старые заявления «T4 не измерялась» относятся к времени тех запусков.

## Зафиксированная модель и результаты

Hugging Face **BART encoder–decoder**, **20,106,240 параметров**: encoder4 слоя, decoder4 слоя, d_model384,6 heads в каждом attention, FFN1536, GELU, learned positional embeddings512, dropout0.1, shared encoder/decoder embeddings и LM head. Архитектура из BartConfig, исходно случайные веса. Никаких внешних pretrained weights. Для инференса from_pretrained(local_files_only=True) читает **собственные** safetensors.

Frozen baseline — artifacts/baseline/model/ и tokenizer.json; полное неизменённое состояние для resume — artifacts/baseline/resume.pt. Это hard links к выбранным ночным весам, без копирования больших файлов. SHA256 checkpoint: **a677bc9d0834907466e20728ef08712ab919c12b61292d57cc5ffe38a5f482c2**. Hashes ключевых входов/конфигов — artifacts/baseline/identity_before.json. Baseline сохранён независимо от выбранного decoding.

**Свежая локальная проверка baseline:**512 всех задач,493 finite scalar references,330 совпадений (**66.94%**),0 синтаксических ошибок,9 runtime/process failures,3 hit_limit; CE на1500 validation=0.2955000124. Это **совпадение численных результатов выполнения с reference-программой**, math.isclose(rel_tol=1e-6,abs_tol=1e-6), **не текстовое совпадение и не официальный pass@1**. References могут быть ошибочны. Generated ошибки остаются нулём, знаменатель493 не меняется; всего19 непригодных references (10 nonnumeric,8 runtime_error,1 no_result).

**Прежние внешние результаты, сообщённые пользователем:** одна T4, greedy generation≈27.31s, весь процесс≈79.46s; public pass@1=**637/960=66.35%**; no code0%, syntax0%, runtime1.56%, wrong answer32.08%. Это не новые измерения и не результаты выбранного beam. Raw outputs/версии того запуска в этом сообщении не предоставлены.

Дневная A_current завершена: CE0.293922,85/121,323/493; generation-best остался исходным330/493. B и main дневной сессии **не обучались**.5.82M cache был подготовлен, но в выбранной модели не использован. Историческая причина wall-clock разрыва не установлена; новый training CLI проверяет time.time непосредственно, в том числе после возобновления системы.

## Данные, фильтры, split, tokenizer

Разрешённый датасет SlavaYus/tinysms_qa_12m, pinned revision b2724689ff079e9c22efd548fe134bfb29036dbb; train11,838,003 строки/60 shards. Raw файлы и download_manifest остаются в data/tinysms_qa_12m. Первоначально5000 случайных индексов из каждого shard, seed42+shard, затем nested expansion до40000/shard. После технических фильтров и точной дедупликации выбрано **2,368,669 пар /2,329,435 нормализованных вопросов**, включая исходные298,011 train-пар. Выбранная линия получила **11,819,810 предъявлений**, а не столько уникальных задач. Не включённые в lineage A_original и day/A_current хранятся как отдельные опыты.

Фильтры: непустые question/code, parseable AST, синхронная callable solve без обязательных аргументов; exact question/code duplicate removal. Числа/пунктуация/код не переписываются. Source=[BOS]+question+[EOS], максимум192; target=code+[EOS], максимум384. Слишком длинная train-программа исключается целиком. Исполнение train не используется для фильтра/новых targets; отрицательные и дробные значения не отбрасываются.

Группа = SHA256(' '.join(question.split())). **Все** группы hash%200==0 навсегда исключены из train, включая не вошедшие в1500 validation. Validation — один неизменённый reference на группу, фиксированные1500 вопросов; generation512 — прежний stratified-by-length набор, включает старые128. Численные ответы references используются только в оценке, не в generation и не в train targets.

Byte-level BPE8192 обучен только на100k training rows, seed42; PAD/BOS/EOS/UNK=0/1/2/3, ByteLevel(add_prefix_space=False), без text normalization. Tokenizer frozen: SHA2564b25bf57faf2b1ce8b143850c12993864cdff7e8ec0d565f1dd7dad466a4d2b0. Первоначальный tokenizer готовился в отдельном environment tokenizers0.22.1; обучение модели и инференс использовали0.20.3. data/processed/training и original ссылаются на сохранённые исторические mmap/parquet caches: нет многогигабайтного дублирования. Для будущей подготовки применяется тот же алгоритм и сохранённые indices; полная подготовка заново в этой сессии не запускалась.

## Проверка выделенного ядра

artifacts/baseline/reproduction/checks.json:179 state_dict entries совпали, embeddings tied, старый/новый greedy идентичен на32 фиксированных задачах; на всех512 новые тексты программ совпали с сохранённым baseline. Маски/PAD/decoder shift и первые20 batches epoch4 идентичны. Содержимое validation и hashes не изменились. Generate получает только question; CE отдельно получает reference tokens для teacher-forcing оценки.

resume_check.json: восстановлены175 optimizer states, scheduler, GradScaler, RNG и sampler(epoch4,batch23069); global step184689, successful optimizer updates184617, LR1e-4. **Ни одного обновления весов не выполнялось.** Проверка не перезаписывала resume.pt. Новые длинные обучения/посылки отсутствуют.

## Ограниченное сравнение decoding

Один checkpoint/tokenizer,512 задач/493 denominator, max_new_tokens256, FP16 autocast/SDPA/KV-cache, do_sample=False, один returned sequence. Для beam search сначала length_penalty1.0, early_stopping=False; повторные ограничения/эвристики не добавлялись. Во всех строках **batch32**. Greedy переиспользован из свежей baseline-проверки, все варианты warmed; время — tokenization/batching/transfers/generation/decode, без загрузки/warmup/исполнения программ. Peak — allocated PyTorch, не физический замер nvidia-smi. Бюджет сравнения60min, фактически≈4min плюс отдельная batch-проверка.

| Режим | Совпадения | Agreement | Syntax errors | Runtime errors* | Hit limit | Mean/p95 tokens | Generation s | Peak GiB | Исправлено/испорчено |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 330/493 | 66.94% | 0 | 9 | 3 | 66.7/122.9 | 8.33 | 0.246 | 0/0 |
| beam2_lp1 | 334/493 | 67.75% | 0 | 10 | 1 | 67.8/124.0 | 39.55 | 0.461 | 14/10 |
| beam4_lp1 | 332/493 | 67.34% | 2 | 10 | 4 | 69.8/129.4 | 60.16 | 0.809 | 18/16 |
| beam2_lp08 | 335/493 | 67.95% | 0 | 9 | 1 | 67.0/121.4 | 36.53 | 0.461 | 14/9 |
| beam2_lp12 | 334/493 | 67.75% | 0 | 10 | 2 | 68.7/124.9 | 36.56 | 0.461 | 13/9 |

*Runtime excludes синтаксические ошибки; статусы nonnumeric/no_result отдельно в JSON. Все отказы остаются нулями. Генерация исполняется только в прежнем pinned Docker: no network/host mounts, UID65534, read-only root, dropped capabilities, no-new-privileges, CPU/RAM/PID/time/output limits.

Выбран **beam2,length_penalty0.8**:335/493 (**67.95%**),+5 задач/+1.01п.п. к greedy,14 исправлено/9 испорчено. Это небольшой локальный выигрыш на одной validation после выбора среди нескольких режимов; уверенного вывода о generalization нет. LP0.8 превосходит LP1.0 всего на1 задачу — **слабое свидетельство**. Два луча против четырёх при LP1.0 отличаются на2 задачи — тоже слабое свидетельство. Публичные вопросы/ответы бота не использовались для выбора режима или программ; test-программы не исполнялись.

Отдельно batch64 проверен без выбора по accuracy: изменились4 beam-программы и2 greedy-программы относительно batch32. Для выбранного beam сохранён batch32; score335/493 относится именно к нему. [deployment_batch_check.json](experiments/decoding/deployment_batch_check.json) не часть чистого сравнения num_beams. Greedy deployment config generation_greedy_t4.json сохраняет прежний batch64, использовавшийся в старом Kaggle notebook; generation_greedy.json с batch32 — для сопоставимой локальной оценки. На T4 beam ещё не измерен и может не уложиться в180s; при превышении вернитесь к MODE="greedy".

## Команды (из корня hw2)

Ничего ниже не запускается автоматически. Существующие данные и baseline уже готовы. Исходные датасеты/checkpoints исключены .gitignore.

```bash
# При необходимости получить/проверить разрешённые raw данные (они уже скачаны)
.venv-experiments/bin/python scripts/download_dataset.py
.venv-experiments/bin/python scripts/verify_dataset.py

# Подготовка в НОВЫЙ каталог; отдельный environment requirements-prepare.txt
.venv-analysis/bin/python -m src.prepare initial --output data/new_initial
.venv-analysis/bin/python -m src.prepare expanded --initial data/new_initial --indices data/processed/training/sampled_row_indices.npz --output data/new_expanded

# Проверка resume без optimizer step, не обучение
.venv-experiments/bin/python -m src.train --resume artifacts/baseline/resume.pt --check-resume

# Если позднее явно потребуется обучение с нуля: случайные веса, готовый разрешённый cache
.venv-experiments/bin/python -m src.train --config configs/baseline.json --output experiments/new_random_run --seconds 1200

# Если позднее явно потребуется resume: прежние optimizer/scaler/scheduler/sampler/RNG
.venv-experiments/bin/python -m src.train --config configs/baseline.json --resume artifacts/baseline/resume.pt --output experiments/new_resume_run --seconds 1200

# Frozen validation: не перезаписывайте уже существующий output
.venv-experiments/bin/python -m src.evaluate --generation-config configs/generation_greedy.json --output experiments/greedy_check_new --ce

# Test generation; собственные локальные веса, без exec generated-кода
.venv-experiments/bin/python -m src.generate --bundle artifacts/baseline --generation-config configs/generation_selected.json --output-dir artifacts/local_selected_test
```

Для новых caches надо указать соответствующие train/validation/tokenizer paths в отдельной конфигурации. Tokenizer mismatch и неявное продолжение с другим dataset fingerprint запрещены. Baseline/существующий output не может стать training output. Constant LR1e-4 в новой training-команде — продолжение выбранной поздней recipe; полный исторический cosine baseline0.0005 хранится в archive и не подменяется этими примерами. Новая training CLI после выделения проверена **на восстановлении**, не полноценным долгим обучением; никакой новый fit в этой задаче не разрешён и не выполнялся.

## Kaggle и что скачать

1. Загрузите kaggle_bundle.zip как **Private Kaggle Dataset**, сохранив структуру папок. Не загружайте train или resume.pt.
2. Импортируйте [notebooks/T4_inference.ipynb](notebooks/T4_inference.ipynb), подключите Dataset и выберите T4. Чистый kernel, первая cell ограничивает CUDA_VISIBLE_DEVICES=0 до CUDA initialization.
3. MODE="selected" — beam2/LP0.8/batch32; MODE="greedy" — прежний deployment greedy/batch64. Одна настройка возвращает baseline. Запустите Run All. Версии/GPU/settings выводятся; weights только local_files_only. Pip может потребовать Internet, CUDA torch предоставляется Kaggle, без Comet/API keys.
4. Notebook измеряет dependency install отдельно, load/warmup/generation и весь дочерний процесс с imports/writes. Warmup не скрывается. Новые времена выбранного beam на T4 и его official pass@1 **не установлены**.
5. Для бота — /kaggle/working/submission.jsonl:1919 физических JSONL строк, ровно одно поле code. Программы не редактируются/не обрезаются;64 токена — только рекомендация. В input JSONL — index/неизменённый question; gold нет. Проверяются count/index/UTF-8 question SHA256.
6. Скачайте **kaggle_results.zip** и выполненный notebook. ZIP содержит submission.jsonl,predictions.json,timing.json,generation_parameters.json,input_alignment.json,submission_validation.json,inference.log. При наличии официального pass1_questions.jsonl тексты сопоставляются и submission явно выстраивается в его порядок; без него этот отдельный порядок не сравнен непосредственно.

Локальное число335/493 — не обещание нового public результата. Ни новых посылок, ни исправления тестовых ответов/новых train targets нет.

## Что читать и сколько строк

Восемь основных Python-модулей, **1457 физических строк** (включая пустые/комментарии; src/__init__.py отдельно1 строка). Код развёрнут в обычное форматирование вместо длинных semicolon-строк.

| Файл | Строк | Функции/роль, порядок чтения |
|---|---:|---|
| [src/prepare.py](src/prepare.py) | 410 | group, is_validation, technical_reason, prepare_initial, prepare_expanded: filters, split, train-only BPE and frozen selection replay |
| [src/data.py](src/data.py) | 97 | TokenDataset, LengthBuckets, collate, train_loader: mmap, order, PAD masks, decoder shift |
| [src/model.py](src/model.py) | 51 | create_model, load_model, verify_model_state: random BART / own local weights |
| [src/train.py](src/train.py) | 284 | restore, save, run: optimizer/scheduler/scaler/RNG/sampler and explicit training CLI |
| [src/generate.py](src/generate.py) | 204 | generate_questions, run: question-only deterministic decoding and inference timings |
| [src/evaluate.py](src/evaluate.py) | 141 | validation_inputs, validation_loss, evaluate_variant: fixed-denominator execution agreement |
| [src/sandbox.py](src/sandbox.py) | 180 | one_batch, execute, compare: pinned restricted Docker, never host execution |
| [src/submission.py](src/submission.py) | 90 | export_submission:1919 JSONL rows, hashes and optional official order |

Дополнительно: configs/baseline.json и generation_*.json; notebooks/T4_inference.ipynb; scripts/verify_core.py (только audit с импортом старого кода для сравнения, не часть pipeline), experiments/decoding/compare.py, scripts/package_project.py (упаковка). Основное ядро не импортирует archive. Старые runner/controllers в archive — только история, не новый актуальный framework.

## Нерешённые ограничения

- Beam/T4 latency180s и official score требуют самостоятельного Kaggle запуска; прежние greedy T4/public результаты сообщены пользователем, здесь не переизмерялись.
- Исходный pass1_questions.jsonl и raw outputs старого T4/бота отсутствуют. Сохраняется исходный bundled test order; есть опциональная точная проверка официального файла.
- Другая GPU/torch/CUDA и batch могут менять floating-point greedy/beam ответы; batch-эффект обнаружен явно. Локальная эквивалентность подтверждена в одном окружении.
- Полная подготовка11.84M заново и новая длительная training-траектория после рефакторинга не запускались; сохранённые caches/weights/RNG/optimizer проверены.
- Docker нужен только локальной оценке. На Kaggle generated-код не исполняется.

Решение сдать до24.10.2026 23:59, PDF-отчёт до25.10.2026 23:59. Текущие Markdown-отчёты не объявляются готовым финальным PDF.

Локальная проверка нового Kaggle-пакета на RTX3060: полный beam-инференс1919 задач131.39s (load0.111s,warmup1.560s,pipeline133.43s без interpreter imports); полный greedy64=19.53s. Все1919 новых greedy64 records/code/token counts совпали с историческим локальным инференсом. Это не новые T4-замеры. Проверка упаковки/JSONL/order и отсутствия изменений baseline — artifacts/project_audit.json. Полные HF defaults и effective параметры вариантов сохранены в configs/generation_hf_defaults.json и */effective_generation_parameters.json.

В review_bundle.zip отсутствуют веса, caches, полная история и validation/test payloads; он предназначен для чтения текущего ядра. Проверка old-vs-new в scripts/verify_core.py требует локальной archive/истории и не включается в компактный review ZIP; её подтверждённые результаты лежат в verification/. Полные исторические материалы сохранены на диске, карта переноса включена.
