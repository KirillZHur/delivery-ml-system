# Защита за 5 минут — Delivery ML System

Показывать **результаты из `reports/report.md` после полного прогона на Mac**. Ниже сценарий и файлы, а не зафиксированные вручную цифры.

| Время | Основной тезис | Доказательство в проекте |
|---|---|---|
| 0:00–0:40 | Срок доставки нужен покупателю при заказе; MAE и недооценка важнее одной красивой оценки, бюджет HTTP p95 ≤100 мс при 20 rps. | `configs/data.json`, `reports/report.md` §1 |
| 0:40–1:30 | Два времени у события: `event_at` и `ingested_at`; оба должны быть ≤ времени прогноза. Метка становится готовой через семь дней + час. | `src/features.py`, `tests/test_point_in_time.py`, `reports/report.md` §2 |
| 1:30–2:30 | Медиана → Ridge → бустинг; модель выбрана на феврале, честно проверена на закрытом марте; улучшение подтверждено bootstrap по дням. Отдельный срез дальних заказов с дождём имеет покрытие ниже 90%. | `reports/training_metrics.json`, `reports/evaluation_metrics.json`, `reports/report.md` §3–4 |
| 2:30–3:30 | Реальный FastAPI, полный HTTP benchmark до/после оптимизации, 20 и 40 rps, p95 включая PIT и сериализацию. Показывать конкретно **машину своего замера**, не выдавать Linux за Mac. | `reports/performance.json`, `reports/latency_raw.csv`, `reports/report.md` §5–6 |
| 3:30–4:30 | Shadow → 5% → 25% → 100%, sticky routing; v2 замедляется, контроллер откатывает; v1 проверена через HTTP. Общая авария признаков требует ремонта источника, а не отката модели. | `reports/rollout_windows.csv`, `reports/rollout_events.jsonl`, `reports/report.md` §7–8 |
| 4:30–5:00 | Модель обновляется только по зрелым меткам; независимый gate допускает хорошего кандидата и отклоняет испорченного. Следующий шаг — A/B-тест и внешний нагрузочный стенд. | `retraining/full_run/decisions/`, `reports/report.md` §9–10 |

## Схема системы

```mermaid
flowchart LR
    A[Синтетический поток] --> B[PIT-признаки]
    B --> C[Train / Validation / Test]
    C --> D[Модель + манифест]
    D --> E[FastAPI]
    E --> F[Shadow / Canary / Rollback]
    F --> G[Мониторинг]
    G --> H[Зрелые когорты]
    H --> I[Переобучение + Gate]
    I --> D
```

## Полезные демонстрации

- **FastAPI Swagger:** `http://127.0.0.1:8000/docs` при запущенном `bash scripts/run_service.sh`.
- **Контроль отката:** строки `candidate_latency`/`candidate_errors` в `reports/rollout_summary.json` и события `reports/rollout_events.jsonl`.
- **Отсутствие будущих данных:** тесты `tests/test_point_in_time.py` и проверка `point_in_time_future_events_used == 0` в `reports/data_summary.json`.
- **Зрелость меток:** `reports/monitor_cohorts.csv` и знаменатель `mae_denominator` в `reports/monitor_summary.json`.
- **Переобучение:** `retraining/full_run/registry.json` и оба решения в `retraining/full_run/decisions/`.

## Вопросы, на которые стоит ответить

- Почему бустинг не выбран? Ridge дал меньшую MAE на независимой валидации; сложность сама по себе не цель.
- Почему нельзя считать незрелую доставку опоздавшей? Это неизвестная метка, а не целевой ноль.
- Почему дрейф не означает ухудшение MAE? Сдвиг входов без зрелых исходов не сообщает величину ошибки.
- Почему fallback и rollback — разные механизмы? Fallback даёт резервный ответ при сбое общего источника; rollback возвращает версию при проблеме кандидата.
- Что реально измерено? HTTP-задержки на одном локальном стенде. Качество — на синтетических метках, инциденты выпуска — симулированы.
