"""Build the final evidence-based technical report directly from generated measurements.

Never transcribe measured numbers manually into Markdown tables. Verify key
artifact contracts and operational gates before claiming a successful run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

from data.generate import ROOT
from src.serving_bundle import load_serving_bundle


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest() -> str:
    digest_object = hashlib.sha256()
    for group in ("src", "data", "configs", "scripts", "tests"):
        for path in sorted((ROOT / group).rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".json", ".sh"):
                continue
            relative = path.relative_to(ROOT).as_posix()
            digest_object.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest_object.hexdigest()


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           text=True, capture_output=True, check=False, timeout=3)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def fmt(number: float, decimals: int = 3) -> str:
    return f"{number:.{decimals}f}"


def check_inputs(profile: str) -> tuple[dict, dict]:
    data = load("reports/data_summary.json")
    training = load("reports/training_metrics.json")
    eval_report = load("reports/evaluation_metrics.json")
    performance = load("reports/performance.json")
    release = load("reports/rollout_summary.json")
    monitor = load("reports/monitor_summary.json")
    good = load("retraining/full_run/decisions/delivery-2026.2-linear.json")
    bad = load("retraining/full_run/decisions/delivery-2026.2-linear-quality-failure.json")
    state = load("retraining/full_run/registry.json")
    model, manifest, serving = load_serving_bundle(ROOT / "artifacts")
    del model, serving
    checks = {
        "source_data_no_future_events": data["point_in_time_future_events_used"] == 0,
        "data_count": data["orders"] == 24000 and data["events"] > 0,
        "source_data_hashes_match": all(
            digest(ROOT / "data/generated" / f"{name}.csv") == value
            for name, value in data["sha256"].items()
        ),
        "features_sha_matches_model": manifest["data_sha256"] == digest(ROOT / "data/generated/features.csv"),
        "validation_selects_linear": training["selected_model"] == "linear",
        "test_holdout_never_used_for_selection": not eval_report["evaluation_policy"]["test_used_in_selection_or_calibration"],
        "strict_test_improvement": eval_report["paired_mae_improvement"]["ci_lower_hours"] > 0,
        "prediction_interval_cover": eval_report["prediction_interval"]["test"]["empirical_coverage"] >= .85,
        "all_required_slices_available": all(s["status"] == "evaluated" for s in eval_report["mandatory_slices"]),
        "service_model_parity": performance["offline_quality_unchanged"]["max_http_parity_absolute_error"] <= 1e-12,
        "meets_latency_20_rps": performance["modes"]["fast"]["steady"]["latency_ms"]["p95_ms"] <= 100,
        "http_no_errors_at_20_rps": all(performance["modes"][mode]["steady"]["errors"] == 0 for mode in ("baseline", "fast")),
        "http_no_degraded_at_20_rps": all(performance["modes"][mode]["steady"]["degraded"] == 0 for mode in ("baseline", "fast")),
        "http_20_rps_sustained": all(performance["modes"][mode]["steady"]["completed_rps"] >= 19 for mode in ("baseline", "fast")),
        "canary_healthy_promoted": next(s for s in release["scenarios"] if s["name"] == "healthy")["status"] == "promoted",
        "canary_technical_error_rolls_back": next(s for s in release["scenarios"] if s["name"] == "candidate_errors")["status"] == "rolled_back",
        "canary_latency_rolls_back": next(s for s in release["scenarios"] if s["name"] == "candidate_latency")["status"] == "rolled_back",
        "canary_small_window_holds": release["small_window"]["action"] == "hold",
        "rollout_restored_v1_http": release["rollback_http"]["status"] == "verified",
        "monitor_clean_no_alert": monitor["clean_alerts"] == [],
        "monitor_data_failure": any(a["type"] == "data_quality" for a in monitor["injected_alerts"]),
        "monitor_persistent_drift": any(a["type"] == "drift" for a in monitor["injected_alerts"]),
        "monitor_mature_denominator": monitor["mature_quality"]["mae_denominator"] == monitor["mature_quality"]["mature_ready_orders"],
        "retraining_good_candidate_passes": good["outcome"] == "promoted" and all(good["checks"].values()),
        "retraining_bad_candidate_rejected": bad["outcome"] == "rejected" and bool(bad["rejection_reasons"]),
        "retraining_active_unchanged_after_rejection": state["active_version"] == good["candidate_version"] == bad["active_version_after"],
        "valid_profile": profile in ("full", "quick"),
    }
    record = {"checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks),
              "all_passed": all(checks.values()), "profile": profile}
    return record, {
        "data": data, "training": training, "evaluation": eval_report,
        "performance": performance, "release": release, "monitor": monitor,
        "retraining_good": good, "retraining_bad": bad, "registry": state,
    }


def reproducibility(profile: str, test_count: int, outcome: dict) -> dict:
    file_groups = ["configs", "data/generated", "artifacts", "releases", "reports", "retraining/full_run"]
    checksums = {}
    for group in file_groups:
        for path in sorted((ROOT / group).rglob("*")):
            if not path.is_file() or path.name in ("report.md", "rubric.csv", "reproducibility.json", "pipeline_summary.json"):
                continue
            checksums[path.relative_to(ROOT).as_posix()] = digest(path)
    result = {
        "profile": profile, "tests_passed": test_count,
        "source_sha256": source_digest(), "git_commit": git_commit(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "files_sha256": checksums, "quality_gates": outcome,
    }
    return result


def rubric_rows() -> list[tuple[str, int, str, str]]:
    return [
        ("Данные и время", 15, "data/generate.py; src/features.py; src/splits.py; tests/test_point_in_time.py; reports/data_summary.json", "контракт, PIT, зрелые метки, тесты"),
        ("Лестница моделей", 10, "src/train.py; reports/training_metrics.json", "median, Ridge, HistGradientBoosting; валидационный выбор"),
        ("Воспроизводимость", 10, "scripts/run_all.sh; src/pipeline.py; requirements.txt; configs/; reports/reproducibility.json", "одна команда, версии, конфигурации, SHA-256, изоляция обновления"),
        ("Офлайн-оценка", 15, "src/evaluate.py; reports/evaluation_metrics.json; reports/evaluation_slices.csv", "MAE, интервалы, срезы, бутстрэп по дням, зрелые исходы"),
        ("Сервис и тесты", 15, "src/service.py; src/feature_store.py; tests/test_service.py; scripts/check_service.sh", "FastAPI, PIT-паритет, timeout, ошибка контракта, резерв"),
        ("Оптимизация", 10, "src/fast_predict.py; src/benchmark.py; reports/performance.json; reports/latency_raw.csv", "реальный HTTP, прогрев, 20/40 rps, точность до/после"),
        ("Выпуск и откат", 10, "src/rollout.py; src/replay.py; reports/rollout_events.jsonl; reports/rollout_windows.csv", "shadow, sticky canary, hold, rollback, HTTP-восстановление"),
        ("Мониторинг", 10, "src/monitor.py; reports/monitor_daily.csv; reports/monitor_cohorts.csv; reports/monitor_alerts.jsonl", "DQA, PSI/KS/chi-square, зрелые когорты"),
        ("Переобучение", 5, "src/retrain.py; retraining/full_run/decisions/; retraining/full_run/registry.json", "новые зрелые метки, отдельный gate, допуск/отказ, предыдущая версия"),
    ]


def generate_markdown(parts: dict, outcome: dict, *, profile: str, test_count: int, code_sha: str) -> str:
    d = parts["data"]; t = parts["training"]; e = parts["evaluation"]
    p = parts["performance"]; r = parts["release"]; m = parts["monitor"]
    good = parts["retraining_good"]; bad = parts["retraining_bad"]
    ci = e["paired_mae_improvement"]; pi = e["prediction_interval"]
    qs = m["mature_quality"]
    rows = []
    add = rows.append
    add("# Итоговый технический отчёт — Delivery ML System")
    add("")
    add(f"**Источник:** машинные результаты полного конвейера, профиль `{profile}`. **Тесты:** {test_count} passed. "
        f"**Контрольный SHA-256 исходников:** `{code_sha}`.")
    add("" )
    add("## 1. Контракт решения и ограничения")
    add("Сервис оценивает число часов доставки в момент оформления заказа. Входы: расстояние (0–1000 км), "
        "известная к моменту решения загрузка склада и прогноз дождя. Выход: оценка, 90%-й интервал, "
        "идентификатор версии и флаг деградации. Метрики: MAE (главная), недооценка и покрытие интервала; "
        "целевой p95 полного HTTP-ответа ≤100 мс на 20 запросах/с. Простое решение — медиана тренировочной выборки.")
    add("Формула синтетического срока: `clip(12 + 0.04·distance_km + 18·warehouse_load + 6·rain_forecast + N(0,3), 1, 168)`. "
        "Преобразования и версия модели загружаются единым проверяемым комплектом.")
    add("")
    add("## 2. Данные, границы и предотвращённые утечки")
    add(f"Генератор `data/generate.py`: seed **{d['seed']}**, **{d['orders']}** заказов, "
        f"**{d['events']}** событий признаков; использование будущих событий: **{d['point_in_time_future_events_used']}**.")
    add("Для выбора события одновременно соблюдаются `event_at ≤ ordered_at` и `ingested_at ≤ ordered_at`. "
        "Поля `order_id`, времена, `delivery_hours` и фактические исходы исключены из матрицы признаков. "
        "Метка становится известной через семь суток и один час. Незрелые метки не считаются нулевыми.")
    add("")
    add("| Период UTC | Размер | Готовые метки | Исключённые незрелые | Снимок |")
    add("|---|---:|---:|---:|---|")
    for name, part in d["split_readiness"].items():
        add(f"| {name}: {part['period_start'][:10]} — {part['period_end'][:10]} | {part['orders_total']} | {part['labels_ready']} | {part['labels_immature']} | {part['snapshot_as_of'][:10]} |")
    add("Состав train/validation/test/replay последовательный по времени. Все данные и метрики имеют контрольные суммы "
        "в `reports/reproducibility.json`; переменный seed не подменяется известными ответами.")
    add("")
    add("## 3. Лестница моделей и протокол выбора")
    add("Препроцессинг обучается только на train. Выбор модели производится по зрелым февральским меткам; мартовский test открыт однократно.")
    add("")
    add("| Модель | MAE train, ч | MAE validation, ч |")
    add("|---|---:|---:|")
    for label, label_display in (("median", "Константная медиана"), ("linear", "StandardScaler + Ridge"), ("hist_gradient_boosting", "HistGradientBoostingRegressor")):
        metric = t["models"][label]
        add(f"| {label_display} | {fmt(metric['train']['mae_hours'])} | {fmt(metric['valid']['mae_hours'])} |")
    add(f"Выбрана **{t['selected_model']}**, потому что дала минимальную валидационную MAE. "
        "Медиана не учитывает признаки, линейная модель выявляет почти линейную зависимость, бустинг здесь не улучшает обобщение.")
    add("")
    add("## 4. Закрытый тест, оценка интервалов и слабые срезы")
    add(f"На **{e['split_counts']['test_mature']}** зрелых мартовских заказах: baseline MAE **{fmt(e['baseline_test']['mae_hours'])} ч**, "
        f"модель MAE **{fmt(e['candidate_test']['mae_hours'])} ч**, "
        f"средняя недооценка кандидата **{fmt(e['candidate_test']['mean_underestimation_hours'])} ч**.")
    add(f"Парное улучшение MAE: **{fmt(ci['estimate_hours'])} ч**, 95%-й bootstrap ДИ "
        f"**[{fmt(ci['ci_lower_hours'])}; {fmt(ci['ci_upper_hours'])}] ч**, "
        f"{ci['repeats']} повторов, единица ресэмплирования — календарный день UTC ({ci['n_blocks']} блока).")
    add(f"90%-й интервал, калиброванный по зрелой валидации ({pi['calibration_residual_count']} строк): "
        f"фактическое тестовое покрытие **{fmt(100 * pi['test']['empirical_coverage'], 2)}%**, "
        f"средняя ширина **{fmt(pi['test']['mean_width_hours'])} ч**.")
    add("")
    add("| Заранее заданный срез | n | MAE модели, ч | 95%-й ДИ улучшения, ч | Покрытие интервала |")
    add("|---|---:|---:|---|---:|")
    for s in e["mandatory_slices"]:
        add(f"| {s['slice']} | {s['n_orders']} | {fmt(s['candidate_mae_hours'])} | "
            f"[{fmt(s['ci_lower_hours'])}; {fmt(s['ci_upper_hours'])}] | {fmt(100*s['empirical_interval_coverage'], 2)}% |")
    add("Слабый срез — дальняя доставка с дождём: его эмпирическое покрытие ниже номинальных 90%. "
        "Этот интервал не имеет строгой conformal-гарантии, поскольку валидация использовалась как для выбора модели, так и для калибровки. "
        "Для улучшения нужны отдельная калибровка и независимая проверка на новых данных.")
    add("")
    add("## 5. Сервис, резервный путь и эксплуатационный контракт")
    add("`src/service.py` реализует `/predict`, `/health/live`, `/health/ready`, `/metrics`, проверяет типы и диапазоны. "
        "`src/feature_store.py` разрешает только своевременно поступившие события и применяет 20-мс тайм-аут. "
        "Нет признака, истёк срок годности или недоступен источник — явный резервный расчёт с `degraded=true` и кодом причины. "
        "На неизвестные поля/обязательные значения API отвечает ошибкой валидации; порядок признаков берётся из манифеста, а не из порядка ключей JSON. "
        "Проверяются недопустимые значения бинарного признака и ограничения числовых полей; реальное хранилище и очередь здесь не развёрнуты.")
    add(f"Паритет HTTP-ответа с офлайн-эталоном: максимальное расхождение **{p['offline_quality_unchanged']['max_http_parity_absolute_error']:.3g} ч**. "
        "Сервис загружает модель один раз и проверяет SHA-256 и версию калибровки.")
    add("")
    add("## 6. HTTP-нагрузка и ускорение")
    add("Машина замера: `" + p["system"]["platform"] + "` (" + p["system"]["python"] + "). "
        "Отдельные одно-процессные Uvicorn-серверы, одинаковые payload и планировщик, прогрев, "
        "persistent connections, измерение от начала HTTP POST до разбора ответа, включая PIT, JSON и audit. "
        "Синтетический локальный поток не эквивалентен внешней сети. Серверный handler измерен отдельно.")
    add("")
    add("| Режим | Нагрузка | Измеренных запросов | Медиана HTTP, мс | p95 HTTP, мс | Выполнено, rps | Ошибки |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for phase in ("steady", "burst"):
        for mode in ("baseline", "fast"):
            metric = p["modes"][mode][phase]
            latency = metric["latency_ms"]
            add(f"| {mode} | {metric['scheduled_rps']:.0f} | {latency['count']} | {fmt(latency['median_ms'])} | "
                f"{fmt(latency['p95_ms'])} | {fmt(metric['completed_rps'], 1)} | {metric['errors']} |")
    before = p["modes"]["baseline"]["steady"]["latency_ms"]["p95_ms"]
    after = p["modes"]["fast"]["steady"]["latency_ms"]["p95_ms"]
    add(f"Оптимизация `StandardScaler+Ridge → pre-computed linear weights`: изменение p95 при 20 rps "
        f"**{fmt(before)} → {fmt(after)} мс** ({fmt((before-after)/before*100,1)}%). "
        f"Размер файла модели **{p['model_artifact_bytes']} байт**; модель и качество не меняются. "
        "Для иных машин показатели могут быть другими; сырые значения в `reports/latency_raw.csv`.")
    if profile == "quick":
        add("**Профиль QUICK:** укороченные замеры предназначены для проверки интеграции. Перед окончательной защитой нужен FULL-прогон.")
    add("")
    add("## 7. Shadow / canary / rollback")
    add("Воспроизведён sticky user_id assignment, теневой режим, шаги 5%, 25%, 100%, "
        "удержание двух здоровых окон, недостаточность данных и откат. Манифесты версий и артефакты проходят проверку совместимости. "
        "Правило технических ошибок: >1% в двух последовательных окнах по ≥1000 запросов кандидата; "
        "независимо контролируются p95 ≤100 мс и доля degraded ≤5%.")
    add("")
    add("| Сценарий | Решение | Окон | Итоговая доля v2 |")
    add("|---|---|---:|---:|")
    for s in r["scenarios"]:
        add(f"| {s['name']} | {s['status']} | {s['windows']} | {s['share']:.0%} |")
    add(f"Малое окно: **{r['small_window']['window_size']}** запросов, кандидат получил "
        f"**{r['small_window']['candidate_requests']}**, решение **{r['small_window']['action']}**. "
        f"Подтверждение восстановления v1 через HTTP: **{r['rollback_http']['status']}**. "
        "Уже принятые v2-запросы завершаются со своей фактической версией после переключения. "
        "Поломка общего источника требует восстановления источника, а не бессмысленной смены модели.")
    add("**Граница доказательств:** p95/ошибки в контроллере заданы синтетическим сценарием. "
        "Реальные измерения HTTP приведены выше, а не подменяют искусственные инциденты.")
    add("")
    add("## 8. Мониторинг и зрелые когорты")
    add(f"В чистом replay-потоке подтверждённых тревог: **{len(m['clean_alerts'])}**. "
        "Монитор применяет немедленные data-quality проверки, статистики PSI/KS/χ² с поправкой Бонферрони "
        "и серию трёх дней-кандидатов. Повторы измерения складской нагрузки учитываются как события источника, "
        "а не как независимые наблюдения каждого заказа.")
    add("")
    add("| Тип тревоги | Дата | Признак | Наблюдения | Действие |")
    add("|---|---|---|---:|---|")
    for a in m["injected_alerts"]:
        add(f"| {a['type']} | {a['day']} | {a['feature']} | {a['observations']} | `{a['action']}` |")
    add(f"На снимке **{qs['snapshot_as_of']}** готовых когорт **{qs['ready_cohorts']}**, "
        f"неполных **{qs['incomplete_cohorts']}**, незрелых **{qs['immature_cohorts']}**. "
        f"MAE = **{fmt(qs['mae_numerator_hours'])} / {qs['mae_denominator']} = {fmt(qs['ready_mae_hours'])} ч**, "
        f"всего {qs['total_orders']} replay-заказов. "
        "Незрелые исходы не входят в знаменатель. Сдвиг входов не считается доказательством ухудшения качества.")
    add("")
    add("## 9. Локальное переобучение и проверка допуска")
    add("Политика: ежемесячно после закрытия семидневных когорт и дополнительного часа поступления меток; "
        "может запускаться при подтверждённом дрейфе или ухудшении зрелой MAE. "
        "Ответственный в реальной системе — владелец ML-модели / дежурный ML-инженер.")
    add(f"На снимке {good['snapshot_as_of']} — обучено на **{good['counts']['training']}** зрелых заказах "
        f"(новых **{good['counts']['new_training']}**), калибровка **{good['counts']['calibration']}**, "
        f"отдельный gate **{good['counts']['gate']}**. "
        "Мартовские метки test не входят в этот цикл.")
    add(f"Действующая модель MAE gate **{fmt(good['model']['active']['mae_hours'], 5)} ч**, "
        f"кандидат **{fmt(good['model']['candidate']['mae_hours'], 5)} ч**, "
        f"покрытие **{fmt(100*good['model']['gate_intervals']['empirical_coverage'], 2)}%**, "
        f"решение **{good['outcome']}**; "
        f"искусственно испорченная версия MAE **{fmt(bad['model']['candidate']['mae_hours'])} ч**, "
        f"решение **{bad['outcome']}**. После отказа активна **{bad['active_version_after']}**.")
    add("Сервис кандидата проверен через FastAPI TestClient (не внешняя нагрузка); для реального выпуска требуется "
        "проверка в том же целевом окружении и подтверждение владельцем. Предыдущий комплект с проверяемой контрольной суммой SHA-256 остаётся "
        "доступным для rollback, работающий процесс должен быть перезапущен.")
    add("")
    add("## 10. Ограничения и дальнейшая проверка")
    add("Данные независимые и синтетические; нет сезонности, логистических пробок, пространственной автокорреляции, "
        "настоящих изменений спроса или стохастических задержек меток. Условная симуляция ошибок не демонстрирует "
        "устойчивость к реальным отказам инфраструктуры. Результаты p95 получены через localhost в одном процессе; "
        "нет нескольких реплик, реальных сетевых задержек и реального 24/7 SLO. Интервал не имеет строгой "
        "distribution-free гарантии. Пользу для покупателей необходимо оценить отдельным A/B-тестом "
        "с бизнес-метриками обещанного срока и жалоб. Следующий опыт: внешний HTTP-нагрузочный стенд "
        "плюс A/B-тест на реальных доставках с независимой калибровочной выборкой.")
    add("")
    add("## 11. Матрица проверки критериев (100 баллов)")
    add("Таблица ниже показывает **выполненные инженерные свидетельства**, а не гарантированную оценку преподавателя. "
        "Баллы по строке — максимальные доступные при принятии преподавателем всех проверок.")
    add("")
    add("| Критерий | Максимум | Свидетельства |")
    add("|---|---:|---|")
    for name, points, evidence, desc in rubric_rows():
        add(f"| {name} | {points} | {desc}; `{evidence.split(';')[0]}` |")
    add(f"| **Итого** | **{sum(row[1] for row in rubric_rows())}** | **{outcome['checks_passed']}/{outcome['checks_total']} автоматических интеграционных проверок** |")
    add("Полный перечень файлов-подтверждений — в `reports/rubric.csv`; SHA-256 всех выходов и версия среды — "
        "в `reports/reproducibility.json`. Для финальной оценки отдельно выполняется полный прогон на целевой машине.")
    add("")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify measured outputs and build report/rubric")
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    existing = ROOT / "reports/pipeline_summary.json"
    profile = args.profile
    if args.verify_only and existing.exists():
        profile = load("reports/pipeline_summary.json").get("profile", "full")
    status, parts = check_inputs(profile)
    failed = [name for name, value in status["checks"].items() if not value]
    if failed:
        raise RuntimeError("Evidence verification FAILED: " + ", ".join(failed))
    print(f"Evidence verification: {status['checks_passed']}/{status['checks_total']} passed", flush=True)
    if args.verify_only:
        return
    test_count = args.test_count
    result = reproducibility(profile, test_count, status)
    reports = ROOT / "reports"
    (reports / "reproducibility.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (reports / "rubric.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("criterion", "maximum_points", "evidence_paths", "verified_scope"))
        for row in rubric_rows():
            writer.writerow(row)
    (reports / "report.md").write_text(generate_markdown(parts, status, profile=profile,
                                   test_count=test_count, code_sha=result["source_sha256"]), encoding="utf-8")
    print("Wrote reports/report.md, reports/rubric.csv, reports/reproducibility.json", flush=True)


if __name__ == "__main__":
    main()
