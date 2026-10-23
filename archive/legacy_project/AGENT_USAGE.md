# Использование агента

Агент сохранил baseline по SHA256, выделил и отформатировал src без замены HF BART/attention/cache, проверил179 tensors, batch/masks/sampler,32 old-vs-new greedy и все512 historical programs,330/493 execution agreement/CE. Проверка resume восстановила175 optimizer states/scheduler/scaler/RNG/step без optimizer step. История перемещена с compatibility symlinks, historical reports не переписывались.

Агент провёл ограниченное decoding-only сравнение greedy/beam2/beam4 и LP0.8/1.2 на frozen validation; выбрал335/493, сохранив baseline330/493. Deployment batch проверялся отдельно, обнаружены численные различия. Public вопросы/эталонные ответы бота не использовались для обучения/выбора программ. Подготовлены src, README, manifests, Kaggle notebook/ZIP и компактный review bundle. Новых targets/внешних pretrained weights/длительных fit/посылок нет. User-provided prior T4/public results документированы как прежние, не как новые измерения.
