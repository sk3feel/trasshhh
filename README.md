# NLP HW2 — baseline и готовый Colab A100-эксперимент

## Что делать сейчас

Большая модель **106 321 920 параметров ещё не обучена**. Новый код обучения находится целиком в корневом `Colab_train_A100.ipynb`; корневой `Colab_inference_T4.ipynb` — независимый инференс. Оба самодостаточны: без git clone, импорта src, загрузки наших скриптов и API keys. Библиотеки устанавливаются обычным pip, Drive подключается с подтверждением пользователя. Новые длительные обучения локально не запущены.

1. Загрузите **`data/colab/colab_training_data.zip` одним ZIP без распаковки** в Google Drive: **`My Drive/nlp_hw2/colab_training_data.zip`**. Это единственный внешний вход нового training notebook. Размер **860 992 113 байт (~861 МБ)**; распакованные данные **1 695 635 353 байт (~1,70 ГБ)**. Рекомендуется дополнительно ≥10 GiB свободного Drive и ≥20 GiB локального диска runtime для данных/optimizer/checkpoints/экспортов.
2. Откройте [A100 notebook в Colab](https://colab.research.google.com/github/sk3feel/trasshhh/blob/colab-large-experiment/Colab_train_A100.ipynb), выберите **A100**. Настройки первой code cell готовы: `DRIVE_ROOT=/content/drive/MyDrive/nlp_hw2`, `DATA_ARCHIVE=DRIVE_ROOT/'colab_training_data.zip'`, новый `RUN_NAME`, `RESUME_FROM=None`. При стандартном пути ничего менять не надо.
3. Сразу нажмите **Runtime → Run all**, подтвердите Drive. Отсчёт начинается в первой code cell; время между подключением GPU и Run all тоже расходует облачный бюджет, но точно не измеряется кодом. Данные копируются/распаковываются в `/content`; train читает локальные mmap, не Drive.
4. Дождитесь `completion.json` и сообщения **DRIVE EXPORTS VERIFIED AND LOADABLE**. Проверьте `My Drive/nlp_hw2/runs/<RUN_NAME>/`. После сообщения **вручную Runtime → Disconnect and delete runtime**. Окончание Python не отключает GPU. Если Drive сообщил ошибку, сначала скачайте локальные `/content/nlp_hw2/runs/<RUN_NAME>/`: успешное сохранение не подтверждено.
5. Скачайте из run **`review_bundle.zip`** для локальной оценки, передайте его локальному Codex. Команда из корня: `.venv-experiments/bin/python archive/colab_preparation/scripts/evaluate_colab_review.py --review /путь/review_bundle.zip --output data/outputs/colab_numeric_review`. Нужны работающий Docker и pinned image из `baseline_66_35/src/sandbox.py`. Результат — `summary.json` и execution/metrics по доступным кандидатам.
6. Откройте [T4 notebook в Colab](https://colab.research.google.com/github/sk3feel/trasshhh/blob/colab-large-experiment/Colab_inference_T4.ipynb) в **чистой T4-сессии**. `EXPORT_PATH=None` берёт CE-экспорт последнего завершённого run. После локальной численной оценки при необходимости укажите `/content/drive/MyDrive/nlp_hw2/runs/<RUN_NAME>/exports/<candidate_name>`. `INFERENCE_BATCH=64` независим от training batch. Run all, подтвердить Drive.
7. Скачайте **`t4_results.zip`** по ссылке последней ячейки или из `My Drive/nlp_hw2/inference/T4_<timestamp>/`. В нём `submission.jsonl`, `timing.json`, `generation_parameters.json`, `predictions.json`, `submission_validation.json`, `inference.log`. Боту предназначен **только submission.jsonl**:1919 строк, в каждой только `{"code":"..."}` в исходном порядке. Боту автоматически ничего не отправляется.
8. Вручную отключите T4 через **Disconnect and delete runtime**. Верните локальному Codex `t4_results.zip` и `review_bundle.zip` для анализа. Ни одна часть T4 notebook не зависит от прежнего A100 `/content`.

Ветка: [colab-large-experiment](https://github.com/sk3feel/trasshhh/tree/colab-large-experiment). Для воспроизводимого запуска можно открывать notebooks по permalink опубликованного commit. Старый commit `f0e25e9` остаётся в истории с прежними `notebooks/…` путями; текущие notebooks перенесены в корень **без изменения байтов**. Их implementation fingerprint сохраняется в результатах вместе с config и data identity.

## Структура

```text
README.md                       основная передача контекста
Colab_train_A100.ipynb           единственная реализация нового106M обучения
Colab_inference_T4.ipynb         самостоятельный T4 inference
baseline_66_35/                 прежний frozen20.1M baseline + минимальный код
data/
  tinysms_qa_12m/               исходный датасет и download manifest
  processed/{original,training,expanded}  указатели на frozen caches
  colab/colab_training_data.zip единственный архив для загрузки на Drive
  colab/prepared/               готовое распакованное содержимое; hard links
archive/
  assignment/                  исходное условие
  experiments/                 все прежние сессии, checkpoint, configs, отчёты
  legacy_project/              прежний README, bundles, проверки, deployment
  legacy_notebooks/            устаревшие notebooks
  colab_preparation/           helper scripts, configs, локальные проверки
  reorganization_20261023/     before/after hashes и аудит переноса
  PATH_MAP.json                старые → новые пути
```

`.git`, `.gitignore` и скрытые `.venv-analysis`/`.venv-experiments` сохранены. Venv не переносились: абсолютные shebang/пути установленного окружения могли бы сломаться. Действующих AGENTS.md в дереве проекта не обнаружено. Корневых compatibility symlinks к старым experiments/src/bundles нет.

Все исходные файлы и checkpoints сохранены. Большие исторические cache/parquet остаются физически в `archive/experiments/…`: смешаны с provenance своих сессий, поэтому массовый перенос мог сломать исторические связи. Понятный рабочий доступ — `data/processed/`; валидные symlinks после переноса проверены и исправлены. Подготовленные Colab массивы уже используют hard links, повторного многогигабайтного копирования не было. Данные, веса, venv и большие логи в Git не добавляются. Полная локальная история богаче компактного GitHub-репозитория.

## Задача, правила и сроки

[Условие](archive/assignment/hw2_code_generation.md): Transformer encoder–decoder с нуля генерирует Python-программу по вопросу. Только разрешённый датасет; запрещены дополнительные данные/targets, дистилляция и внешние pretrained weights. Все1919 тестовых ответов требуется генерировать ≤180 секунд на **одной NVIDIA T4**. Решение: **24.10.2026 23:59**, отчёт PDF: **25.10.2026 23:59**. Сдать весь использованный код, inference notebook и отчёт с экспериментами/использованием агентов; без отчёта оценка обнуляется. Public pass@1>0.33 — порог задания, private leaderboard влияет на место.

Бот принимает JSONL по одной программе на вопрос в порядке `pass1_questions.jsonl`; числовой результат исполнения сравнивается с эталоном при rtol=atol1e-6. Рекомендация **64 токена не является обязательным ограничением**: в проверенном pipeline и новом согласованном плане max_new_tokens256. Generated-код не редактируется для получения правильных ответов. Тестовые reference-ответы отсутствуют в пакетах, test не используется для выбора модели.

## Архитектуры и данные

| Вариант | Параметры | Encoder/decoder | d / heads / FFN | Статус |
|---|---:|---|---|---|
| Исходный baseline |20 106 240|4+4|384 /6 /1536|Frozen, сохранён отдельно|
| Ранняя large40m |40 453 120|См. исторический config|См. config|Короткий неудачный опыт|
| Overnight C |34 148 352|4+4|512 /8 /2048|Обучена с нуля,314/493|
| Новый Colab A100 |106 321 920|6+6|768 /12 /3072|**Не обучена**, только подготовлена|

HF BART, исходно random initialization; собственные сохранённые модели читаются `from_pretrained(local_files_only=True)` без скачивания pretrained. Общие настройки: learned positions512, dropout0.1, shared encoder/decoder embedding и tied LM head. Byte-level BPE8192 обучен **только на100k training rows**, seed42; normalization отсутствует, ByteLevel(add_prefix_space=False). PAD/BOS/EOS/UNK0/1/2/3. Tokenizer frozen SHA256 `4b25bf57faf2b1ce8b143850c12993864cdff7e8ec0d565f1dd7dad466a4d2b0`.

Датасет **SlavaYus/tinysms_qa_12m**, revision `b2724689ff079e9c22efd548fe134bfb29036dbb`:11 838 003 train-строки,60 shards. Начально5000 случайных индексов/shard, seed42+shard; вложенное расширение40000/shard дало2 368 669 train-пар; затем до100000/shard либо весь короткий shard —5 820 047 пар. Frozen sample indices/provenance сохранены. Многомиллионный train не загружается в Python-список: NumPy mmap + массив порядка индексов.

Фильтры: непустые question/code, корректный AST, синхронная callable `solve` без обязательных аргументов; exact question/code duplicates удаляются. Source `[BOS]+question+[EOS]`≤192, target `code+[EOS]`≤384; длинные targets исключаются целиком. Числа, punctuation и программы не переписываются. Execution не применяется для train-фильтрации/новых targets. Отрицательные/дробные числа не отбрасываются.

Группа: `SHA256(' '.join(question.split()))`. **Все** группы `hash%200==0` навсегда исключены из train, даже если не вошли в1500 teacher-forcing validation. Generation512 неизменна и включает старые128;493 references дают конечное scalar-число,19 — непригодны. Generated ошибки остаются нулями в493 знаменателе. Численные reference-значения не подаются генератору.

Упаковщик повторно проверил manifests и пересчитал группы всех5 820 047 provenance-вопросов: пересечения с reserved holdout нет. Точный набор и порядок в ZIP закреплены SHA; перенос не менял ZIP. Архив SHA256 `6eed1ad2118aaa0aa7d3c600902101a2c83034d902d9cb21efc9e42fd2933f15`. Notebook проверяет SHA архива/manifest/files, tokenizer, длины и исключение групп.

## История экспериментов

Все training measurements ниже — **локальная RTX3060 12GB**, не A100/T4. Время округлено; разные исторические таблицы измеряют wall/active time по-разному — детали в исходных отчётах. «Локально» — численное reference agreement, **не официальный pass@1**; CE — teacher-forcing loss. Итоговые выводы ограничены одним seed, повторным выбором по одной validation и шумными references.

| Дата/этап | Train / предъявления | Минут | CE | Локально | Вывод |
|---|---|---:|---:|---|---|
|21–22.10 initial20M, LR5e-4|298011 /868753|19.34|0.4491|60/121|Рабочая исходная точка|
|21–22.10 LR1e-3|298011 /231808|9.13|1.1632|0/121|Коллапс генерации, досрочная остановка|
|21–22.10 large40.5M|298011 /152960|8.95|1.1947|2/121|Досрочная остановка; не равный бюджет с baseline|
|22.10 night A_original|298011 /1031249|26.23|0.41591|278/493;69/121|Сравнение исходного train|
|22.10 night B_expanded|2368669 /1053405|26.27|0.38397|262/493;68/121|Короткий trial не предсказал позднее качество|
|22.10 main_continuation|2368669 /6903482|181.34|0.31362|309/493;82/121|Продолжение расширенной ветки|
|22.10 late_continuation LR1e-4|2368669 /2994170|76.37|0.29550|**330/493**;85/121|Выбран исходный frozen baseline|
|22.10 day A_current|2368669 /1473565|36.30 active/perf|0.29392|323/493 final;330 best исходный|B/main не обучались, timeout/перерыв; cache5.82M подготовлен|
|22–23.10 decoding fixed weights|Без обучения|≈4 +batch-check|Та же модель|greedy330, beam2 LP1=334, beam4=332, beam2 LP0.8=**335**, LP1.2=334 из493|Выбран beam2 LP0.8; небольшой validation gain|
|23.10 overnight A_current|2368669 /1757021|54.47|0.29283|**332/493**|Малое улучшение против330|
|23.10 overnight B_expanded|5820047 /2382095|54.43|0.28816 final|**326/493** final;330 best исходный|CE улучшился, численное качество хуже|
|23.10 overnight C_large_scratch34.1M|2368669 /6431674|238.49|0.29708|**314/493**|Дан полный4-часовой бюджет; baseline уже долго обучался|
|23.10 overnight D_continue_A|2368669 /6763991|149.97|0.28770|**333/493**|Лучший новый greedy, **не заменяет baseline_66_35**|
|23.10 новый A100106M|План5820047|Не запускался|Неизвестно|Неизвестно|Два коротких LR trial, затем одна main ветка|

Sanity/overfit и smoke — технические проверки, не модельные победы: высокий LR давал низкий loss и одинаковые программы; более длинный sanity при5e-4 восстановил16/16 train-программ. Resume smoke и маленькие CPU-проверки не считать validation/public.

Источник подробностей: [initial report](archive/experiments/session_20261021/REPORT.md), [night report](archive/experiments/night_20261022/REPORT.md), [day recovery](archive/legacy_project/RECOVERY_STATUS.md), [overnight report](archive/experiments/overnight_20261023/REPORT.md), [прежний README с decoding](archive/legacy_project/README.before_reorganization.md). Исторические отчёты **не переписаны**: например, day report старый и неполный; более поздние state/summary/recovery уточняют завершение. Старые команды требуют перевода путей по [PATH_MAP](archive/PATH_MAP.json), их работоспособность без адаптации не обещается.

## Локальные и официальные метрики: разные свидетельства

| Вариант | Локальный reference agreement | CE | Public, сообщённый пользователем | T4, сообщённый пользователем |
|---|---|---|---|---|
|Frozen greedy20M|330/493, batch32|0.2955000124|**637/960=66,35%**|generation27.31s/process79.46s, прежний greedy batch64|
|Те же веса, beam2 LP0.8|335/493, batch32|Тот же checkpoint|**644/960=67,08%**|generation110.59s/subprocess179.82s|
|Overnight D20M|333/493|0.28770275|Неизвестно|Не измерялось|
|Новая A100106M|Не измерялось|Не измерялось|Неизвестно|Не измерялось|

Локальные hashes/checkpoints/CE/generations/результаты исполнения подтверждаются файлами. Public и приведённые T4-времена **сообщены пользователем**, здесь заново не измерены. Оригинальные scored-посылки и сырые timing-файлы этих T4-запусков не найдены. Baseline веса подтверждены исходным manifest и совпадением state tensors; public association отдельно документирована как user-reported, без ложного доказательства идентичности посылки.

Baseline resume SHA256 `a677bc9d0834907466e20728ef08712ab919c12b61292d57cc5ffe38a5f482c2`; новый333 checkpoint SHA256 `03f31e6f018e4e3f7700acd31703a0ffac3ea17360d0a050a90ebb32c9e79e7f`. Это разные артефакты. [Baseline README](baseline_66_35/README.md) содержит происхождение, local inference/evaluate/resume-check команды и отдельный greedy Kaggle notebook.

## Новый A100 план, бюджет и сохранение

**Всего облако≤6,5 часа: A100≤6 часов, отдельно T4≤30 минут.** В A100 включены Drive mount/зависимости/copy/extract/проверки/benchmark/trials/main/validation/save/export. Ориентир: setup+benchmark≤20мин, из них benchmark≤10мин; два LR trial вместе≤40мин; main≤4ч30мин; последние30мин без updates. Общий deadline важнее этапов: задержки сокращают main. Код не прекращает оплату runtime; отключение только вручную, без keep-alive. Зависшее Drive I/O/внезапная потеря VM не дают гарантии сохранения.

A100 проверяется по GPU name/compute capability и BF16 support. BF16 autocast, SDPA, AdamW/clipping, без compile, CUDA PyTorch из Colab автоматически не заменяется. Effective batch256; real physical32/64/128/256 с accumulation8/4/2/1. Benchmark — отдельные discardable модели с одинаковыми initial weights/RNG; нормальные и длинные блоки, warmup отдельно, target tokens/s/examples/s/update seconds/allocated/reserved. Выбирается устойчивый быстрый вариант с20% memory margin. Прогноз эпохи и предъявлений рассчитывается по замеру, не является гарантией.

Accumulation нормирует суммарный градиент по непустым target-токенам полного update, включая разные длины и неполный последний блок. LR2e-4/5e-4: одинаковые случайные weights/RNG/order/effectivebatch/updates, короткий warmup, затем постоянный peak. Автовыбор — меньший token-weighted CE среди стабильных полностью завершённых trial. Если оба неуспешны, диагностическая остановка без новых гипотез.

Main стартует **с исходных random weights**, не продолжает trial. Planned steps фиксируются один раз по оставшемуся времени/benchmark/оценочным накладным расходам; warmup3%, cosine до0.1peak. CE примерно каждые25мин, fixed512 greedy примерно каждый час при наличии бюджета. Раннее отставание от долго обучавшегося baseline не причина остановки. Colab не исполняет generated-программы. CE выбирает кандидатов лишь предварительно; численная accuracy и public пока неизвестны.

Drive `runs/<RUN_NAME>/`: `configs/`, `logs/`, `checkpoints/`, `validation_predictions/`, `exports/`, `review_bundle.zip`, `completion.json`, `budget.json`. Полный last содержит model/optimizer/scheduler/RNG/sampler position/счётчики/config/data+code identity; BF16 без GradScaler. До трёх best-CE кандидатов и final — отдельные inference weights. Запись сначала локально, затем Drive temp, SHA-проверка, смена указателя. Предыдущая исправная full-копия сохраняется до успешной публикации новой. Сообщение об отключении A100 выдаётся только после проверки и загрузки экспортов непосредственно с Drive.

После обрыва откройте **чистую A100-сессию**, в первой cell задайте прежний `RUN_NAME` и `RESUME_FROM=Path('/content/drive/MyDrive/nlp_hw2/runs/<RUN_NAME>/checkpoints/last.json')`. При явно повреждённом указателе — `last.previous.json`. Benchmark/LR не повторяются; проверяются config/identity/schedule, восстанавливаются состояния и sampler. Budget сохраняет исходный абсолютный deadline: **новых6 часов нет**, downtime считается консервативно потраченным. После исходного cutoff продолжение не разрешается. Между GPU/версиями битовая идентичность не гарантируется.

## Три разных вида архивов

| Артефакт | Назначение | Содержимое |
|---|---|---|
|`colab_training_data.zip`|Вход A100, загрузить один раз на Drive|Train token caches, frozen validation, tokenizer, manifests, reference artifacts; без weights/test answers|
|`review_bundle.zip`|Скачать после A100 для локального анализа|Configs/logs/validation generations/references/metadata; **без train и weights**|
|Inference bundle / `exports/<candidate>`|Перенос на T4|Own model.safetensors/config, tokenizer, generation config, standalone inference code/requirements/manifest, неизменные1919 questions; **без optimizer/train**|

Только численная локальная команда вызывает прежнюю Docker-песочницу: pinned Python image, без сети/bind mounts, read-only root, UID65534, capabilities removed, лимиты CPU/RAM/PID/time/output. Все512 выходов сохраняются; ошибки generated остаются нулём на493 finite references. Кандидат без полного512 generation явно пропускается. `summary.json` ранжирует по local agreement, CE используется при равенстве; Drive-экспорты автоматически не переключаются. Выбранный путь можно вручную передать T4.

T4 использует FP16/SDPA/KV-cache/greedy/max256 с отдельным inference batch. Измеряет setup/copy, load, warmup, full generation, полный pipeline отдельно с CUDA synchronization. Проверяются1919 outputs/индексы/question hashes. Прямое сравнение с отдельно загруженным официальным `pass1_questions.jsonl` в новом notebook не выполняется: сохраняется закреплённый исходный bundled-порядок. Окончательные180s и качество106M не заявляются до измерения.

## Проверки, агенты и открытые вопросы

До переноса малые **CPU** проверки определений из основного notebook прошли: syntax, config equality, masks/decoder shift, token-weighted accumulation, partial block, optimizer update, exact next-update resume с dropout, запрет смены schedule, export/reload и одинаковые tiny greedy outputs, deadline без update и сохранение последнего Drive-pointer при copy failure. Это не Run all в реальном Colab. После переноса обучение для проверки не запускается; проверяются notebook bytes/syntax/imports/paths/manifests и frozen weights. Доказательства — `archive/colab_preparation/docs/` и `archive/reorganization_20261023/`.

**Не проверено:** реальный чистый Colab Run all на A100/T4, текущие Torch/CUDA/Transformers compatibility, GPU full-state resume, BF16 throughput и лучший batch, доступная RAM/VRAM/диск, скорость/сбои Drive, качество новой106M модели, её T4 generation≤180s и official pass@1. CPU-тест не доказывает стабильность большого GPU fit. Expanded train5.82M не гарантирует улучшения: в коротком late-resume B качество ухудшалось при меньшем CE; это не доказательство вреда данных для новой большой модели. Небольшой gain333 против330 не имеет статистической гарантии.

Агенты использовались для исследования условия, загрузки/аудита данных, кода random-BART обучения/инференса/песочницы, запуска ограниченных локальных серий, анализа ошибок, выделения прежнего ядра, сравнения decoding, упаковки, самодостаточных Colab notebooks и проверок. Это влияло на выбор tokenizer/batch/LR/данных и диагностику collapse. Историческое описание — [AGENT_USAGE](archive/legacy_project/AGENT_USAGE.md); новые notebooks подготовлены агентом, но большой A100 fit ещё не выполнялся. Новые targets агент не создавал, test-программы для выбора не исполнял, посылки боту не отправлял.

**Не менять:** frozen baseline/checkpoint, tokenizer, validation/split/order, исходные данные и исторические отчёты. Новые outputs — только в новые каталоги; новые гипотезы/обучения требуют отдельного указания. Старые baseline и результаты не перезаписываются новым333/493 или будущим106M экспортом.
