# Ночная сессия: продолжение baseline и расширение данных

Старт: 2026-10-22T00:29:13+00:00, жёсткое завершение: 2026-10-22T06:29:13+00:00; training cutoff: 2026-10-22T06:09:13+00:00 (последние 20 минут — сохранение/оценка). Время до генерации отчёта: 5.00 h. Europe/Moscow = UTC+3. Все GPU-замеры локальные, RTX 3060 12 GB; замера T4 нет.

## Ответы на основные вопросы

1. **Улучшение относительно 60/121.** Выбранный checkpoint: 85/121 на старых 128 против 60/121 исходного baseline; изменение +25 совпадений.
2. **Расширенная validation.** На фиксированных 512: 330/493 (66.94%). Baseline на том же наборе: 240/493 (48.68%).
3. **Эффект train expansion.** После сопоставимых коротких продолжений A: 278/493, B: 262/493; B−A = -3.25 п.п. Это один seed, шумные references и ограниченный набор. Долгое продолжение получает только одна ветка: его дальнейший выигрыш нельзя приписать исключительно расширению данных, отделив от дополнительного времени обучения.
4. **Переобучение/нестабильность/плато.** Устойчивого роста validation CE не наблюдалось: основная фаза снизила CE с 0.384 до 0.314, поздняя — до 0.2955. На 512 временно замедлился рост (295→298→297→300), но затем получены 309 и 330; окончательное плато не установлено. Коллапса генерации не видно: все 512 программ различаются и ответы чувствительны к входу. Пропуски AMP редкие, не серии неудачных обновлений. Коллапс, overflow и train/validation тренды проверялись отдельно. Разрыв train/validation сам по себе не доказывает переобучение; учитывать динамику CE и расширенного agreement на кривых.
5. **Перенос на T4.** Кандидат для переноса — `selected/best_generation.pt` (полное состояние); для инференса — `t4_bundle/`. Веса выбраны по фиксированным 512, CE-checkpoint сохранён отдельно. Перед первой официальной посылкой проверить на одной T4, воспроизводимость и формат файла бота. Посылки не отправлялись.
6. **Следующее действие.** Сначала воспроизвести inference на T4 и подтвердить 180 s, затем проверить официальный формат и сделать первую посылку самостоятельно. По диагностике выбранной модели agreement падает от 115/126 на самых коротких вопросах до 57/119 на самых длинных; это корреляция, не доказательство причины. Если потребуется дальнейшее обучение, наиболее понятный следующий опыт — добавить ещё разрешённые train-строки с теми же tokenizer/holdout и LR=1e-4, сравнив с продолжением на текущих 2.37M при равном бюджете; не увеличивать архитектуру без отдельной гипотезы. При плато улучшение локальной проверки/анализ ошибок может быть полезнее повторения тех же строк.

Все reference agreement — шумные локальные proxy, не официальный pass@1. Validation используется только для оценки и выбора checkpoint, не для новых train targets. Маленькие различия не являются уверенным улучшением. Лучшие snapshots выбраны по validation, поэтому оценка имеет selection bias; приватный тест не изучался.

## Запуски и фактические объёмы

| Запуск | Данные | Минут | Доступно уникальных пар | Предъявлений в фазе | Final CE | Old 128 agreement | New 512 agreement |
|---|---|---:|---:|---:|---:|---:|---:|
| Original baseline | original | — | 298,011 | — | 0.4491 | 60/121 | 240/493 |
| A_original | original | 26.2 | 298,011 | 1,031,249 | 0.4159 | 69/121 | 278/493 |
| B_expanded | expanded | 26.3 | 2,368,669 | 1,053,405 | 0.3840 | 68/121 | 262/493 |
| main_continuation | expanded | 181.3 | 2,368,669 | 6,903,482 | 0.3136 | 82/121 | 309/493 |
| late_continuation | expanded | 76.4 | 2,368,669 | 2,994,170 | 0.2955 | 85/121 | 330/493 |

Таблица показывает **конец каждой фазы**, а выбранный лучший snapshot может быть раньше. `experiments.csv/json` содержат successful updates, skip count, nonpad source/target tokens, фактически увиденные уникальные строки, вопросы, VRAM и скорость. Сумма предъявлений по завершённым новым фазам (без smoke): 11,982,306. Эта сумма включает расход на обе экспериментальные ветки; это не объём обучения одного выбранного checkpoint. Его lineage/counters сохранены в `selected/selection.json` и checkpoint. Исходный baseline до этой ночи получил 868,753 предъявления.

`unique_seen_including_original_pretraining` учитывает, что исходный baseline уже прошёл все 298,011 старых пар; новые пары подсчитываются по точному membership marker. Доступные строки не смешиваются с повторными предъявлениями. При расширении меняется набор и его sampler, не tokenizer/validation.

## Проверка resume и расписание

Исходный optimizer LR: 0.0001561835; cosine last_epoch=13574, horizon=20000. GradScaler: {"scale": 1048576.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000, "_growth_tracker": 1526}; пропусков до ночи 2/13576. Сохранены веса, optimizer moments, scaler и RNG. Для обеих A/B веток **осознанно перезапущен scheduler на constant LR=0.00015**, близкий к фактическому source LR; он не зависит от размера датасета/длины эпохи. Эффективный и GPU batch=64, accumulation=1. Самплер обнулён в обеих ветках для контролируемого старта; старый offset новой выборке не применялся. Дальнейшие продолжения сохраняют sampler position, scheduler и optimizer; снижение LR, если было, записано отдельной фазой.

`runs/resume_smoke/resume_smoke_verified.json` подтверждает изменение весов, реальные optimizer steps, ожидаемый LR, конечный loss и рост счётчиков; новый last.pt повторно прочитан. Smoke не служит исходной точкой A/B: обе ветки стартуют с неизменённого original best.pt. Его SHA-256: `0e160915bbf613c1e9858d8c9464477eeb453648c0c7be92f26ca04eb6aa7a41`. Полный audit: `artifacts/resume_audit.json`.

## Данные

Expanded: **2,368,669 уникальных question/code пар**, **2,329,435 уникальных нормализованных вопросов**. Все 298,011 старых train-пар сохранены. Из всех 60 разрешённых shards добраны индексы до 40000 на shard (короткий — целиком). Точные индексы и question/code/shard/row/pair hash сохранены в `data/expanded`. Дедупликация на диске SQLite, токены memmap uint16; никаких миллионов Python-объектов одновременно. Подготовка: 228.4 s, peak RSS 1569.3 MiB. Фильтры: `{"sampled_rows": 2398003, "hash_holdout": 11918, "train_rows": 2368669, "target_over_384": 211, "exact_duplicate_pair": 17104, "source_over_192": 101}`.

Сохраняется старый SHA256(' '.join(question.split())) holdout: все hash % 200 == 0 исключены, правило проверено на всех expanded rows. Числа/пунктуация/question/code не переписывались. Tokenizer frozen BPE8k, SHA256 `4b25bf57faf2b1ce8b143850c12993864cdff7e8ec0d565f1dd7dad466a4d2b0`. Validation parquet SHA256 `4f59340fb14a854facb17857d6e64528605ef5f148148a03acd981a3008e37ab`. Минимальные технические фильтры прежние, лимиты source192/target384 с исключением целых слишком длинных targets. Исполнение миллионов train-программ не выполнялось; execution-based фильтра и новых targets нет.

## Фиксированная оценка

CE по тем же 1500 validation-вопросам, усреднённая по числу непустых target-токенов. Old128 сохранён точно. New512: 10 равных по числу строк страт по длине входа, все old128 включены, остальные выбраны seed 4200+bin без учёта исполнения/ответов. References посчитаны один раз исключительно для оценки. Знаменатели: old128=121, new512=493. Reference statuses: `{"ok": 493, "nonnumeric": 10, "runtime_error": 8, "no_result": 1}`; исключённые индексы/причины — `artifacts/validation_report.json`.

Generated ошибки никогда не уменьшают знаменатель. Сохраняются syntax, execution-completed и finite numeric runtime success, EOS, hit limit, mean/p95 lengths, число уникальных программ и CE sensitivity к перемешанным вопросам. Сравнение чисел `math.isclose(rel_tol=1e-6,abs_tol=1e-6)`. Разрешены отрицательные/дробные; None/nonnumeric/nonfinite/runtime/timeout разделены. Sandbox: прежний pinned Docker, UID65534, без сети/host mounts, read-only root, no capabilities, ограничения CPU/RAM/PID/вывода/времени. Сторонние библиотеки не добавлялись.

## Кривые и сохранение

![Кривые обучения и оценки](curves.png)

Standalone: `curves.png`, `curves.pdf`; исходные точки — `runs/*/metrics.jsonl`. Ось X — минуты **внутри фазы**, линии различных фаз не являются одинаковыми по общей истории обучения. В `runs/*/progress.json` — компактное состояние для продолжения чата.

Для каждой полноценной фазы сохранены только last.pt, best_ce.pt, best_gen.pt; совпадающие snapshots используют hard links. Полное состояние содержит optimizer/scaler/scheduler/RNG, global counters, sampler position и seen bitmap. Выбранный generation snapshot и отдельный CE snapshot — `selected/`; исходные результаты прошлой сессии не перезаписывались. Выбор основан на generation512, а не только CE. Если CE и generation расходятся, оба кандидата отдельно оценены в `evaluations/selected_*`.

## Автономность и команды

`runner.py` — последовательный локальный план без LLM/API-вызовов, с тайм-аутами, общим deadline, сохранением stdout/stderr, traceback и безопасным переходом к финализации вместо бесконечных повторов. Первый A bootstrap зарегистрирован runner-ом и ожидался до запуска B. Все последующие GPU-тренировки запускаются по одной. Этапы: A/B25min → перспективная ветка до3h/плато → при наличии времени одно позднее продолжение (либо тот же LR, либо отдельное снижение до1e-4). Фактические решения/изменения — `state.json`; ошибки — `[]`.

Из каталога hw2, отдельная фаза воспроизводится так:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-experiments/bin/python experiments/night_20261022/engine.py train --config experiments/night_20261022/configs/B_expanded.json --seconds 1500
```

Не запускать повторно в существующий completed run: скрипт это запрещает. Для нового запуска скопировать config и задать новое name. `resume`, LR/restart_sampler/restart_scheduler и data_dir явно записаны в каждом config. Night runner имеет фиксированные абсолютные даты и завершённые run names: для новой ночи нужен новый каталог/новый session.json.

Пересборка отчёта без обучения: `.venv-experiments/bin/python experiments/night_20261022/report.py`. Переносимый инференс: `t4_bundle/generate.py --questions PATH --output predictions.json --batch-size 64`. Это загрузка **собственных обученных локальных весов**, не внешней предобученной модели.

## Полный локальный inference

Все 1919 вопросов в исходном порядке: **19.58 s**, GPU NVIDIA GeForce RTX 3060, batch 64, mean tokens 62.06, p95 116.0, limit fraction 0.47%. Load 0.51 s, warmup 1.25 s, tokenization 0.150 s. Это warmed generation, включая batching/decoding; старт Python/imports отдельно не измерялся. T4 ещё не измерялась. `inference/full_test_timing.json`, `inference/predictions.json`, переносимый `t4_bundle/`. Формат для бота необходимо подтвердить до посылки; отправок не было.

## Фактически изменённые настройки и критерии

Архитектура во всех ночных фазах неизменна: d_model=384, encoder/decoder=4/4, heads=6, FFN=1536, dropout=0.1, vocab=8192, позиции512. Исходные optimizer moments/GradScaler сохранены. Одинаковое для A/B изменение — старый cosine заменён на constant LR=0.00015, sampler reset. Поля `warmup_steps`/`schedule_steps`, унаследованные в JSON исходного запуска, не используются новым constant scheduler; фактические LR проверены по optimizer state и логам. Единственное различие A/B — доступный train. Поздний LR, если изменялся, указан отдельной фазой и config. Оценка расширена до512; checkpoint выбирается по expanded generation, CE-кандидат хранится отдельно. Gradient accumulation=1, GPU/effective batch64; source192, train target384, generation256. torch.compile и сложные kernels не использовались. Peak allocated относится к аллокатору PyTorch, не к отдельно измеренному физическому пику VRAM.
Выбранный generation checkpoint: CE=0.2955, old128=85/121, new512=330/493. Отдельный CE checkpoint: CE=0.2955, old128=85/121, new512=330/493. Для кода основной критерий — совпадение результата, поэтому более низкий CE сам по себе не заменяет generation-оценку; малую разницу нельзя уверенно объявлять улучшением.

Парные изменения на одних и тех же вопросах (без формального вывода о значимости): `{"A_vs_B": {"denominator": 493, "both_match": 227, "only_left": 51, "only_right": 35, "neither": 180}, "baseline_vs_selected": {"denominator": 493, "both_match": 222, "only_left": 18, "only_right": 108, "neither": 145}}`. References шумные; выбранный checkpoint оптимизировался по этой validation.

Исходный best.pt не изменён: True.

## Генерация выбранного checkpoint

| Набор | Syntax | Исполнение завершено | Конечное число | EOS | Лимит | Mean / p95 tokens | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| old128 | 128/128 | 127/128 | 121/128 | 126/128 | 2/128 | 70.8 / 126.0 | 85/121 |
| new512 | 512/512 | 503/512 | 493/512 | 509/512 | 3/512 | 66.7 / 122.9 | 330/493 |

new512 generated execution statuses: `{"ok": 493, "nonnumeric": 8, "no_result": 2, "runtime_error": 8, "process_error": 1}`.


Завершённое исполнение включает nonnumeric/no_result; конечное число не доказывает правильность. Точные reference exclusions сохранены независимо от модели.

Переносимый `t4_bundle/generate.py` повторно выполнен локально на всех 1919 вопросах: все records/code/token counts совпали с основным скриптом в исходном порядке. Проверка: `artifacts/portable_export_verified.json`. Ноутбук `t4_bundle/T4_inference.ipynb` проверен синтаксически; запуска на T4 не было.

Диагностика по квартилям длины вопроса (символы): `[{"question_chars_min": 51, "question_chars_max": 135, "examples": 128, "numeric_references": 126, "matches": 115}, {"question_chars_min": 136, "question_chars_max": 178, "examples": 128, "numeric_references": 122, "matches": 82}, {"question_chars_min": 179, "question_chars_max": 234, "examples": 128, "numeric_references": 126, "matches": 76}, {"question_chars_min": 236, "question_chars_max": 585, "examples": 128, "numeric_references": 119, "matches": 57}]`. Полные неизменённые reference/generated программы ошибок — `selected/error_analysis.json`; только оценка, не train targets. Один process_error с кодом -9 соответствует зациклившейся generated-программе, остановленной лимитом sandbox; её считают неуспехом.

Поздняя фаза: агент после замедления expanded generation выбрал один опыт с LR=0.0001. Файл `supervised_overrides.json` явно заменяет автоматическое предложение runner до старта поздней фазы. Фактические настройки сохранены в `configs/late_continuation.json`, `runs/late_continuation/intent.json`, measured LR и `artifacts/late_supervised_decision.json`; исходный optimizer/scaler/sampler сохранён. Это сравнение последовательных фаз с дополнительным временем, а не причинная оценка LR при равном бюджете.

## Проверки устойчивости по фазам

- A_original: LR=0.00015; CE начальная 0.4491, лучшая 0.4159, конечная 0.4159; 512 agreement 240 → 278; skipped 8/16115 (0.050%); CE shuffled−correct=1.476; уникальных программ на 128 — 128; остановка `time_budget`.
- B_expanded: LR=0.00015; CE начальная 0.4491, лучшая 0.3840, конечная 0.3840; 512 agreement 240 → 262; skipped 7/16460 (0.043%); CE shuffled−correct=1.321; уникальных программ на 128 — 128; остановка `time_budget`.
- main_continuation: LR=0.00015; CE начальная 0.3840, лучшая 0.3133, конечная 0.3136; 512 agreement 262 → 309; skipped 43/107868 (0.040%); CE shuffled−correct=1.290; уникальных программ на 128 — 128; остановка `time_budget`.
- late_continuation: LR=0.0001; CE начальная 0.3136, лучшая 0.2955, конечная 0.2955; 512 agreement 309 → 330; skipped 20/46785 (0.043%); CE shuffled−correct=1.315; уникальных программ на 128 — 128; остановка `time_budget`.

## Финальная проверка сохранения

Сессия закончена примерно за 5 часов, все тренировки остановлены. `artifacts/final_verification.json` подтверждает отсутствие training-процессов, полное состояние checkpoint, SHA-256 кода/весов/входов и одинаковый snapshot для лучших CE и generation. Выбранная линия получила 11,819,810 предъявлений, включая исходный baseline; доступно/увидено 2,368,669 уникальных train-пар. Полная ночная стоимость всех веток — 11,982,306 предъявлений, без smoke. Переносимый bundle — 81.0 MB. Окончательный код/configs сохранён в `code_snapshot_final.tar.gz`; начальный snapshot сохранён отдельно.
