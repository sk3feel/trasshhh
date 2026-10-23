# Дополнительная сессия: текущий train против расширения

Старт 2026-10-22T08:52:51+00:00; максимум до 2026-10-22T11:52:51+00:00; training cutoff 2026-10-22T11:32:51+00:00. Отчёт через 0.38 h. Все замеры — RTX 3060 12 GB, один seed. Статус `A_current`, finished=False; ошибки `[]`.

Результат выбранных весов: **85/121 на old128, 330/493 (66.94%) на fixed512**, CE=0.295500. До этой сессии: 85/121, 330/493, CE=0.295500. Это шумный reference agreement, не официальный pass@1. Все512 включают старые128; выбор лучшего по этой validation даёт selection bias. Никаких официальных посылок и замера на T4.

| Фаза | Уникальных доступно | Минут | Предъявлений в фазе | Final CE | Old128 | Fixed512 |
|---|---:|---:|---:|---:|---:|---:|
| Night best | 2,368,669 | — | — | 0.2955 | 85/121 | 330/493 |

## Данные и неизменные условия

Новая выборка: 5,820,047 уникальных exact question/code пар, 5,635,052 групп. Все2,368,669 прежних train-пар сохранены (disk-backed indexed join). SHA256 нормализованного вопроса %200==0 полностью исключается из train; правила и tokenizer/validation прежние. По100k индексов из каждого из60 shards или весь короткий shard; прежние40k включены. Точные индексы/provenance/SQL dedup/memmap — `data/expanded/`. Исполнение train и новые/переписанные targets не применялись. Counts: `{"sampled_rows": 5938003, "hash_holdout": 29696, "train_rows": 5820047, "source_over_192": 209, "exact_duplicate_pair": 87529, "target_over_384": 522}`. Подготовка 525.6s, peakRSS 2617MiB.

BART20.1M, d384,4+4слоя,6heads, FFN1536, shared embeddings, train-only frozen BPE8192, source192/target384/generation256; greedy FP16 AMP + KV-cache, training FP16 GradScaler + SDPA + AdamW. Только разрешённый train, исходно случайная инициализация. Продолжаем собственный night checkpoint, не внешние pretrained weights. Scheduler constant1e-4 без restart; optimizer/scaler/RNG сохраняются; A/B sampler reset, main sampler продолжается. Других гиперпараметров не меняли.

Сумма предъявлений всех новых фаз: 0; это не число уникальных примеров и не одна линия обучения. Seen bitmap учитывает ранее предъявленные2,368,669 пар; в CSV отдельно available/unique_seen_since_branch_start/unique_seen_including_original_pretraining. Counters выбранной линии — `selected/selection.json`.

## Проверки устойчивости


## Метрики выбранных весов

| Набор | Syntax | Completed execution | Finite numeric | EOS | Cap | Mean / p95 tokens | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| old128 | 128/128 | 127/128 | 121/128 | 126/128 | 2/128 | 70.8/126.0 | 85/121 |
| fixed512 | 512/512 | 503/512 | 493/512 | 509/512 | 3/512 | 66.7/122.9 | 330/493 |

Generated statuses: `{"ok": 493, "nonnumeric": 8, "no_result": 2, "runtime_error": 8, "process_error": 1}`. Frozen reference statuses: `{"ok": 493, "nonnumeric": 10, "runtime_error": 8, "no_result": 1}`. Generated errors всегда считаются неуспехом и не меняют знаменатель493/121. Финитное число не доказывает математическую правильность. Sandbox — прежний pinned Docker без сети/host mounts, UID65534, read-only, capability drops, CPU/RAM/PID/time/output limits; references исключительно evaluation-only.

Парные изменения: `{}`. Полные reference/generated ошибки — `selected/error_analysis.json`, только оценка, не targets.

## Сохранение и воспроизведение

Полные last/best_ce/best_gen по каждой фазе содержат model/optimizer/scaler/scheduler/RNG, sampler и seen bitmap; hard links ограничивают дубли. Старые сессии не перезаписаны. `selected/best_generation.pt` выбран по fixed512, CE сохраняется отдельно, source330 включён среди кандидатов. `state.json`, `logs/`, configs/intent/metrics — история. `curves.png/pdf` — train/val/generation против минут внутри фаз; это разные lineage.

Новый запуск: скопировать config, задать новое name и resume; engine запрещает перезапись completed run. В engine используется абсолютный session cutoff: для другого дня создать новый каталог/session.json.

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv-experiments/bin/python experiments/day_20261022/engine.py train --config experiments/day_20261022/configs/A_current.json --seconds 2100
```

Завершённую фазу повторно в тот же каталог не запускать. Пересобрать отчёт без обучения: `.venv-experiments/bin/python experiments/day_20261022/report.py`. Зависимости: environment-freeze.txt.

Исходный night checkpoint не изменён (SHA256 проверен). Следующий шаг — T4/официальная оценка; новые train-эксперименты выбирать по контролируемому A/B и кривым.
