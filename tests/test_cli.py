"""Tests for the mleval CLI, including end-to-end runs on the example data."""

import json
from pathlib import Path

import pytest

from ml_evaluation_toolkit import cli

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"
BINARY = str(EXAMPLES / "binary_predictions.csv")
MULTICLASS = str(EXAMPLES / "multiclass_predictions.csv")
REGRESSION = str(EXAMPLES / "regression_predictions.csv")


def test_classify_text_output(capsys):
    cli.main(["classify", BINARY])
    out = capsys.readouterr().out
    assert "accuracy" in out
    assert "Confusion matrix" in out
    assert "0.840" in out  # accuracy on the binary example


def test_classify_json_output(capsys):
    cli.main(["classify", MULTICLASS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["accuracy"] == pytest.approx(0.683, abs=0.001)
    assert payload["macro"]["f1"] == pytest.approx(0.623, abs=0.001)
    assert len(payload["per_class"]) == 3
    assert payload["labels"] == [0, 1, 2]


def test_classify_reports_low_recall_class(capsys):
    cli.main(["classify", MULTICLASS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    by_label = {r["label"]: r for r in payload["per_class"]}
    assert by_label[2]["recall"] == pytest.approx(0.359, abs=0.001)


def test_calibrate_text_output(capsys):
    cli.main(["calibrate", BINARY])
    out = capsys.readouterr().out
    assert "ECE" in out
    assert "Brier" in out
    assert "temperature" in out
    assert "over-confident" in out


def test_calibrate_json_output(capsys):
    cli.main(["calibrate", BINARY, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ece"] == pytest.approx(0.091, abs=0.001)
    assert payload["calibrated_ece"] < payload["ece"]
    assert payload["temperature"] > 1.0


def test_calibrate_without_proba_exits_2(tmp_path, capsys):
    path = tmp_path / "p.csv"
    path.write_text("y_true,y_pred\n0,0\n1,1\n")
    with pytest.raises(SystemExit) as exc:
        cli.main(["calibrate", str(path)])
    assert exc.value.code == 2


def test_calibrate_no_temp_flag(capsys):
    cli.main(["calibrate", BINARY, "--json", "--no-temp"])
    payload = json.loads(capsys.readouterr().out)
    assert "temperature" not in payload


def test_errors_text_output(capsys):
    cli.main(["errors", BINARY, "--slice-col", "group"])
    out = capsys.readouterr().out
    assert "1 -> 0: 16" in out
    assert "south" in out
    assert "north" in out


def test_errors_slice_shows_subgroup_gap(capsys):
    cli.main(["errors", BINARY, "--slice-col", "group", "--json"])
    payload = json.loads(capsys.readouterr().out)
    slices = {r["slice"]: r for r in payload["slices"]["slices"]}
    assert slices["south"]["error_rate"] > slices["north"]["error_rate"]
    assert slices["south"]["error_rate"] == pytest.approx(0.257, abs=0.001)


def test_errors_hard_examples_present(capsys):
    cli.main(["errors", MULTICLASS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["hard_examples"]
    assert payload["confusion_pairs"][0]["true"] == 2
    assert payload["confusion_pairs"][0]["predicted"] == 1
    assert payload["confusion_pairs"][0]["count"] == 18


def test_report_binary_full(capsys):
    cli.main(["report", BINARY, "--slice-col", "group"])
    out = capsys.readouterr().out
    assert "Classification report" in out
    assert "Error analysis" in out
    assert "Calibration report" in out


def test_report_json_structure(capsys):
    cli.main(["report", MULTICLASS, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"classification", "errors", "calibration"}
    assert payload["errors"]["error_rate"] == pytest.approx(0.317, abs=0.001)


def test_regress_text_output(capsys):
    cli.main(["regress", REGRESSION, "--slice-col", "group"])
    out = capsys.readouterr().out
    assert "MAE" in out
    assert "R2" in out
    assert "rural" in out
    assert "urban" in out


def test_regress_json_values(capsys):
    cli.main(["regress", REGRESSION, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["mae"] == pytest.approx(10.396, abs=0.001)
    assert payload["rmse"] == pytest.approx(14.143, abs=0.001)
    assert payload["r2"] == pytest.approx(0.989, abs=0.001)
    assert payload["mape_excluded_zero_true"] == 2


def test_report_auto_detects_regression(capsys):
    cli.main(["report", REGRESSION, "--slice-col", "group"])
    out = capsys.readouterr().out
    assert "Regression report" in out
    assert "Classification report" not in out


def test_report_jsonl_input(tmp_path, capsys):
    path = tmp_path / "p.jsonl"
    path.write_text(
        '{"y_true": 0, "y_pred": 0, "y_proba": 0.2}\n'
        '{"y_true": 1, "y_pred": 1, "y_proba": 0.9}\n'
        '{"y_true": 1, "y_pred": 0, "y_proba": 0.4}\n'
    )
    cli.main(["classify", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["accuracy"] == pytest.approx(2 / 3)
