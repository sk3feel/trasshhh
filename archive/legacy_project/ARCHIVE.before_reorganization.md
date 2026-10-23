# Historical material

experiments/ preserves all three prior sessions, original logs/configs/reports/checkpoints/caches and their internal layout. Old experiments/<session> paths are compatibility symlinks. See PATH_MAP.json. Historical prose is unchanged and may describe the older state of the project.

deliveries_before_core_refactor/ preserves both earlier delivery folders and ZIP archives; infrastructure/ preserves their original builder/check scripts. Current code is ../src, current deliveries ../kaggle_bundle.zip and ../review_bundle.zip. No historical training/controller should be started automatically. Large raw data remains under ../data, compact historical caches are reused by data/processed symlinks.
