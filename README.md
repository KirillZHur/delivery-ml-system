# Delivery ML System — прогноз срока доставки

Учебная end-to-end ML-система: синтетические заказы → point-in-time-признаки → обучение и контроль качества → FastAPI → HTTP-нагрузка → shadow/canary/rollback → мониторинг → проверяемое обновление модели. Все инциденты выпуска симулируются, но сервис и HTTP-бенчмарк работают локально. Итоговые цифры и таблицы формируются кодом в [reports/report.md](reports/report.md); карта доказательств для критериев — [reports/rubric.csv](reports/rubric.csv).

## Задача

Пользователь — покупатель маркетплейса в момент оформления заказа. Решение — ожидаемый срок доставки в часах и 90%-й прогнозный интервал. Главная метрика — MAE по **зрелым фактическим срокам**; дополнительная — недооценка срока. Ограничение сервиса: **p95 полного ответа ≤100 мс при 20 запросах/с**. Простая альтернатива — медиана длительности на train. Контракт входа, выбранная модель и границы значений заданы в [configs/data.json](configs/data.json), [configs/training.json](configs/training.json) и [src/service.py](src/service.py).

## Данные

[Генератор](data/generate.py) фиксирует `seed=2026`, создаёт 24 000 заказов (январь–апрель 2026) и журнал событий загрузки склада/прогноза дождя. Расстояние — 0–1 000 км, нагрузка склада — Beta(2, 2), вероятность дождя — 0,27. Срок: `clip(12 + 0.04 × distance_km + 18 × warehouse_load + 6 × rain_forecast + N(0,3), 1, 168)` часов. Скрытый шум, метка, идентификаторы и времена не входят в признаки модели. В журнале хранится момент события и момент его поступления; [src/features.py](src/features.py) допускает признак только при **обоих** временах не позже прогноза. Поздно поступившие события не исправляют задним числом историю. Метка доступна через семь суток после заказа плюс час на доставку данных; незрелые исходы исключаются, см. [src/splits.py](src/splits.py) и [reports/data_summary.json](reports/data_summary.json).

## Запуск

Проверено на Python 3.13; необходим **Python 3.11+**. Фиксированные библиотеки — [requirements.txt](requirements.txt). Из корневой папки проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Полное воспроизведение с реальным HTTP-бенчмарком и всеми тестами
bash scripts/run_all.sh

# Короткий прогон того же конвейера (уменьшен только объём HTTP-нагрузки)
bash scripts/run_quick.sh

# Повторить тесты и сверить артефакты
bash scripts/verify_project.sh
```

Все команды выполняются без изменения Python-кода. Полный прогон формирует/обновляет [reports/pipeline_summary.json](reports/pipeline_summary.json), [reports/reproducibility.json](reports/reproducibility.json), [reports/report.md](reports/report.md) и таблицы. `run_quick.sh` не заменяет полноценные замеры для итоговой сдачи. Хеш исходного дерева фиксируется даже без Git; если папка является Git-репозиторием, также пишется SHA коммита. Скрипт полного прогона перегенерирует результаты и **сбрасывает только демонстрационный** реестр `retraining/full_run/`; другие локальные реестры не трогает.

```bash
# Отдельно поднять стабильный FastAPI (после полного запуска)
bash scripts/run_service.sh
# В другом терминале:
bash scripts/check_service.sh
# Swagger: http://127.0.0.1:8000/docs
# Метрики: http://127.0.0.1:8000/metrics
```

Для работающего сервиса из обновлённого реестра: `DELIVERY_REGISTRY_DIR="$(pwd)/retraining/full_run" bash scripts/run_active_service.sh` (загружает версию на старте; после смены указателя требуется перезапуск). Стандартный сервис и активный сервис используют порт 8000 — одновременно на одном порту не запускаются.

## Обучение

Последовательные временные части: январь — train, февраль — validation, март — **единственный закрытый test**, апрель — последующий replay. Числа созревших меток и исключённых строк — [reports/data_summary.json](reports/data_summary.json). В [src/train.py](src/train.py) по единой MAE на validation сравниваются три ступени: медиана, `StandardScaler + Ridge`, `HistGradientBoostingRegressor`. В [artifacts/](artifacts/) находятся модель, её преобразования, манифест с версией, контрольными суммами и точным порядком признаков. Численные результаты всех трёх моделей — [reports/training_metrics.json](reports/training_metrics.json).

## Оценка

[src/evaluate.py](src/evaluate.py): парный bootstrap разности MAE по блокам UTC-дней (1999 повторов), результат на зрелом закрытом test, четыре заранее определённых среза из [configs/evaluation.json](configs/evaluation.json), недооценка доставки и 90%-й прогнозный интервал. Интервал настроен по зрелой валидации, покрытие проверено на test; строгая conformal-гарантия не заявляется, поскольку validation участвовала в выборе модели. Полные числа, счётчики и интервалы — [reports/evaluation_metrics.json](reports/evaluation_metrics.json), [reports/evaluation_slices.csv](reports/evaluation_slices.csv).

## Производительность

[src/benchmark.py](src/benchmark.py) запускает реальный Uvicorn и генерирует одинаковый поток запросов через localhost для двух путей: исходный `StandardScaler+Ridge` и ускоренная свёртка коэффициентов [src/fast_predict.py](src/fast_predict.py). Измеряются **полный HTTP-ответ** и отдельно обработчик, после прогрева — 20 rps и 40 rps. Проверяются число ошибок, `degraded`, паритет прогнозов, медиана, p95, размер модели, память. Машина замера, версии, сырые задержки — [reports/performance.json](reports/performance.json) и [reports/latency_raw.csv](reports/latency_raw.csv). На другой машине латентность может отличаться; 40 rps — короткая проверка всплеска, а не оценка предельного throughput.

## Выпуск

[src/replay.py](src/replay.py) и [src/rollout.py](src/rollout.py) реализуют shadow → устойчивое назначение пользователей в canary (5%, 25%, 100%) → удержание или откат. Отказ общего источника удерживает долю и включает резерв, индивидуальный сбой кандидата запускает rollback к проверенному комплекту `v1`. Ошибки >1% подтверждаются в двух последовательных окнах при ≥1 000 запросов `v2` на каждое. Задержка и degraded контролируются отдельно; малое окно не позволяет повысить долю. События, окна и контроль фактически обслужившей версии: [reports/rollout_events.jsonl](reports/rollout_events.jsonl), [reports/rollout_windows.csv](reports/rollout_windows.csv), [reports/rollout_summary.json](reports/rollout_summary.json). Числа задержек в симуляторе **искусственные**, а реальный HTTP сравнивается отдельно.

## Эксплуатация

[src/monitor.py](src/monitor.py): немедленные проверки контракта данных и технических ошибок, PSI/KS/χ² для дрейфа с поправкой Бонферрони, серия трёх дней, отдельный MAE **только по готовым когортам**. Данные с повторяемыми значениями загрузки склада не трактуются как независимые измерения источника. Контрольный инцидент содержит отказ `warehouse_load` и сдвиг расстояния. [reports/monitor_alerts.jsonl](reports/monitor_alerts.jsonl), [reports/monitor_daily.csv](reports/monitor_daily.csv), [reports/monitor_cohorts.csv](reports/monitor_cohorts.csv), [reports/monitor_summary.json](reports/monitor_summary.json).

[src/retrain.py](src/retrain.py) выполняет цикл по зрелым данным с последовательным выделением новых train, калибровки и независимого gate. Сравнивает кандидата с текущей моделью по качеству, срезам, покрытию, контракту и HTTP-проверке. Успешный кандидат может сменить активный указатель, заведомо плохой — отклоняется с неизменной рабочей версией. Восстановление возможно через прошлый комплект. Свидетельства — `retraining/full_run/decisions/`, `retraining/full_run/events.jsonl` и `retraining/full_run/registry.json`. Для реальной эксплуатации решение о замене принимает владелец модели.

## Ограничения

Синтетические независимые заказы без сезонности, логистических эффектов и реального потока исходов; фактическая точность и эксплуатационная польза на покупателях не доказаны. Canary-инциденты и задержки в контроллере смоделированы; только HTTP-бенчмарк измеряет работающий сервис на локальной машине. Нет продакшен-инфраструктуры, нескольких реплик и непрерывного внешнего контроля. Оценка интервалов эмпирическая; для строгой гарантии нужна отдельная калибровка. Перед реальным выпуском необходимы внешний нагрузочный тест, тесты реальных задержек исходов и A/B-эксперимент для покупательских метрик. Подробный разбор и таблица по критериям — [reports/report.md](reports/report.md), [reports/rubric.csv](reports/rubric.csv).
