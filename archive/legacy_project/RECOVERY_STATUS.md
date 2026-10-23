# После падения VS Code

Активных experiment/training/runner процессов нет; ничего не останавливали и не перезапускали. Завершена только day/A_current:36.30мин по perf_counter,1473565 предъявлений, CE0.2939217, old128=85/121, fixed512=323/493. Её last.pt/best_ce.pt/best_gen.pt, config/intent/resume_details/metrics/predictions/summary сохранены. Best-generation — начальный night checkpoint330/493, не новое улучшение. BestCE хуже по generation и не выбран.

Расширенный train5,820,047 подготовлен; B_expanded и main_continuation не стартовали. Runner после возвращения управления обнаружил истёкший session cutoff, не начал B, не смог выполнить финальную оценку/отчёт и завершился с TimeoutError('Session reserve reached'). Day selected links/selection.json успели сохраниться; day REPORT был промежуточным. Ночной отчёт и результаты сохранены.

Лог A: command_start09:02:40UTC, command_end15:57:27UTC22.10.2026, при active perf_counter2178сек. Причина разрыва не установлена: приостановка системы/VM возможна, но по логам не доказана. Гарантировать соблюдение трёхчасового host wall-clock в этой ситуации нельзя; новых длительных обучений после восстановления не было. Время Europe/Moscow=UTC+3.

Для первого Kaggle inference выбран неизменённый night/selected/best_generation.pt:85/121,330/493, CE0.295500. Отдельные checkpoint hashes и export comparison — verification/. Полная компактная инвентаризация — RECOVERY_STATUS.json.
