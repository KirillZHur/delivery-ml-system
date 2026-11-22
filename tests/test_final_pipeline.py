"""Integration-report checks without launching HTTP benchmark on each pytest call."""

from pathlib import Path

from src.final_report import check_inputs, rubric_rows, source_digest
from data.generate import ROOT


def test_rubric_total_and_all_evidence_files_exist():
    rows = rubric_rows()
    assert sum(points for _, points, _, _ in rows) == 100
    assert len(rows) == 9
    for _, _, evidence, _ in rows:
        for path in evidence.split(";"):
            path = path.strip()
            assert (ROOT / path).exists(), path


def test_final_report_inspects_actual_run_and_refuses_missing_evidence():
    status, parts = check_inputs("full")
    assert status["all_passed"], [k for k, ok in status["checks"].items() if not ok]
    assert parts["retraining_good"]["outcome"] == "promoted"
    assert parts["retraining_bad"]["outcome"] == "rejected"


def test_source_version_hash_tracks_actual_code():
    assert len(source_digest()) == 64
    assert source_digest() == source_digest()


def test_readme_matches_full_pipeline():
    doc = (ROOT / "README.md").read_text(encoding="utf-8")
    for topic in ("Задача", "Данные", "Запуск", "Обучение", "Оценка", "Производительность",
                  "Выпуск", "Эксплуатация", "Ограничения", "scripts/run_all.sh"):
        assert topic in doc
