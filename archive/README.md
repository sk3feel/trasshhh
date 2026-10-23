# Архив проекта

Действующая инструкция: [корневой README](../README.md). Новый106M код обучения находится только в [корневом A100 notebook](../Colab_train_A100.ipynb); исходный frozen baseline — [baseline_66_35](../baseline_66_35/README.md).

- experiments/: все старые сессии, configs, checkpoints, метрики и неизменные отчёты.
- legacy_project/: прежние README/AGENT_USAGE/recovery, review/kaggle bundles и проверки.
- legacy_notebooks/: прежний notebook с выбором beam/greedy.
- colab_preparation/: helper scripts/configs/docs; нужны только локальным инструментам, не Run All.
- reorganization_20261023/: аудит переноса и read-only проверки; обучение не запускалось.
- assignment/: исходное условие.
- PATH_MAP.json: старые и новые пути, включая сохранённые data pointers.

Корневые compatibility links убраны. Старые тексты/команды исторические: после переноса не обещается работа без адаптации путей. Ничего автоматически не запускать. Большие исторические caches и checkpoints остаются локально и в Git не публикуются.
