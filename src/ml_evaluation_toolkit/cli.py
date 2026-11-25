"""Command line interface: mleval classify | calibrate | errors | report | regress.

Reads saved predictions from a CSV or JSONL file with y_true and
y_pred columns (plus optional y_proba or proba_0..n probability
columns). Output is a human-readable report by default, or JSON with
--json. Output ordering is deterministic: labels sorted, pairs and
slices sorted by count/rate.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import calibration, errors, metrics, regression
from .io import load_predictions


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _print_json(payload):
    print(json.dumps(_jsonable(payload), indent=2))


def _fmt(value, ndigits=3):
    if value is None:
        return "-"
    return f"{value:.{ndigits}f}"


def _is_regression_data(y_true, y_pred):
    """Float targets with non-integer values are regression predictions."""
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return bool(np.any(t != np.round(t)) or np.any(p != np.round(p)))


def _classify_text(summary, pairs=None):
    lines = ["Classification report", "====================="]
    lines.append(f"n = {sum(c['support'] for c in summary['per_class'])}")
    lines.append(f"accuracy          {_fmt(summary['accuracy'])}")
    lines.append(f"balanced accuracy {_fmt(summary['balanced_accuracy'])}")
    lines.append(f"macro F1          {_fmt(summary['macro']['f1'])}")
    lines.append(f"weighted F1       {_fmt(summary['weighted']['f1'])}")
    lines.append(f"MCC               {_fmt(summary['mcc'])}")
    if "roc_auc" in summary:
        lines.append(f"ROC-AUC (macro)   {_fmt(summary['roc_auc']['roc_auc'])}")
    if "average_precision" in summary:
        lines.append(f"avg precision     {_fmt(summary['average_precision']['average_precision'])}")
    lines.append("")
    lines.append(f"{'label':>12} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>8}")
    for row in summary["per_class"]:
        lines.append(
            f"{str(row['label']):>12} {_fmt(row['precision'], 3):>10} "
            f"{_fmt(row['recall'], 3):>8} {_fmt(row['f1'], 3):>8} {row['support']:>8}"
        )
    lines.append("")
    lines.append("Confusion matrix (rows = true, columns = predicted)")
    labels = summary["labels"]
    header = " " * 12 + "".join(f"{str(l):>8}" for l in labels)
    lines.append(header)
    for label, row in zip(labels, summary["confusion_matrix"]):
        lines.append(f"{str(label):>12}" + "".join(f"{v:>8}" for v in row))
    return "\n".join(lines)


def _calibrate_text(summary):
    lines = ["Calibration report", "=================="]
    lines.append(f"ECE        {_fmt(summary['ece'])}")
    lines.append(f"Brier      {_fmt(summary['brier_score'])}")
    lines.append(f"log loss   {_fmt(summary['log_loss'])}")
    if "temperature" in summary:
        lines.append(f"temperature (fitted) {_fmt(summary['temperature'])}")
        lines.append(f"calibrated ECE        {_fmt(summary['calibrated_ece'])}")
        lines.append(f"calibrated Brier      {_fmt(summary['calibrated_brier_score'])}")
        lines.append(f"calibrated log loss   {_fmt(summary['calibrated_log_loss'])}")
    lines.append("")
    lines.append(f"{'bin':>14} {'n':>5} {'mean pred':>10} {'observed':>9} {'gap':>7}  verdict")
    for b in summary["reliability"]["bins"]:
        lo, hi = b["bin"]
        label = f"[{lo:.1f}, {hi:.1f}]" if hi >= 1.0 else f"[{lo:.1f}, {hi:.1f})"
        lines.append(
            f"{label:>14} {b['count']:>5} {_fmt(b['mean_predicted']):>10} "
            f"{_fmt(b['observed_frequency']):>9} {_fmt(b['gap']):>7}  {b['verdict']}"
        )
    return "\n".join(lines)


def _errors_text(summary, slices=None):
    lines = ["Error analysis", "=============="]
    lines.append(f"total {summary['total']}  errors {summary['errors']}  "
                 f"error rate {_fmt(summary['error_rate'])}")
    lines.append("")
    lines.append("Top confusion pairs (true -> predicted)")
    for pair in summary["confusion_pairs"]:
        lines.append(f"  {pair['true']} -> {pair['predicted']}: {pair['count']} "
                     f"(examples {pair['example_indices']})")
    lines.append("")
    lines.append("Error rate by true class")
    for row in summary["error_rate_by_true_class"]:
        lines.append(f"  {row['label']}: {_fmt(row['error_rate'])} "
                     f"({row['errors']}/{row['support']})")
    if "hard_examples" in summary and summary["hard_examples"]:
        lines.append("")
        lines.append("Most confidently wrong examples (highest loss first)")
        for ex in summary["hard_examples"]:
            lines.append(
                f"  row {ex['index']}: true={ex['true']} pred={ex['predicted']} "
                f"P(true)={_fmt(ex['prob_true'])} loss={_fmt(ex['loss'])}"
            )
    if slices:
        lines.append("")
        lines.append(f"Error rate by {slices['slice_column']} "
                     f"(overall {_fmt(slices['overall_error_rate'])})")
        for row in slices["slices"]:
            lines.append(f"  {row['slice']}: {_fmt(row['error_rate'])} "
                         f"({row['errors']}/{row['count']})")
    return "\n".join(lines)


def _regress_text(summary, slices=None):
    lines = ["Regression report", "================="]
    lines.append(f"n    {summary['count']}")
    lines.append(f"MAE  {_fmt(summary['mae'])}")
    lines.append(f"RMSE {_fmt(summary['rmse'])}")
    lines.append(f"R2   {_fmt(summary['r2'])}")
    if summary["mape"] is None:
        lines.append("MAPE - (all true values are zero)")
    else:
        lines.append(f"MAPE {_fmt(summary['mape'])} "
                     f"({summary['mape_excluded_zero_true']} zero true values excluded)")
    if slices:
        lines.append("")
        lines.append(f"Residuals by {slices[0] if slices else ''}")
        for row in slices[1]:
            lines.append(
                f"  {row['slice']}: MAE {_fmt(row['mae'])} RMSE {_fmt(row['rmse'])} "
                f"mean residual {_fmt(row['mean_residual'])} (n={row['count']})"
            )
    return "\n".join(lines)


def _load(args):
    df, y_true, y_pred, y_proba, labels = load_predictions(args.input)
    return df, y_true, y_pred, y_proba, labels


def cmd_classify(args):
    _, y_true, y_pred, y_proba, labels = _load(args)
    summary = metrics.classification_summary(y_true, y_pred, y_proba, labels)
    if args.json:
        _print_json(summary)
    else:
        print(_classify_text(summary))


def cmd_calibrate(args):
    _, y_true, _, y_proba, labels = _load(args)
    if y_proba is None:
        print("error: calibrate needs probabilities (y_proba or proba_0..n columns)",
              file=sys.stderr)
        raise SystemExit(2)
    summary = calibration.calibration_summary(
        y_true, y_proba, n_bins=args.bins, labels=labels, fit_temp=not args.no_temp
    )
    if args.json:
        _print_json(summary)
    else:
        print(_calibrate_text(summary))


def cmd_errors(args):
    df, y_true, y_pred, y_proba, labels = _load(args)
    summary = errors.error_summary(y_true, y_pred, y_proba, labels,
                                   top_n_pairs=args.top_n, top_n_hard=args.top_n)
    slices = None
    if args.slice_col:
        slices = errors.slice_error_report(df, args.slice_col)
    if args.json:
        payload = dict(summary)
        if slices:
            payload["slices"] = slices
        _print_json(payload)
    else:
        print(_errors_text(summary, slices))


def cmd_regress(args):
    df, y_true, y_pred, _, _ = _load(args)
    summary = regression.regression_summary(y_true, y_pred)
    slices = None
    if args.slice_col:
        rows = regression.residual_by_slice(df, args.slice_col)
        slices = (args.slice_col, rows)
    if args.json:
        payload = dict(summary)
        if slices:
            payload["slices"] = {"slice_column": slices[0], "rows": slices[1]}
        _print_json(payload)
    else:
        print(_regress_text(summary, slices))


def cmd_report(args):
    df, y_true, y_pred, y_proba, labels = _load(args)
    if _is_regression_data(y_true, y_pred):
        summary = regression.regression_summary(y_true, y_pred)
        slices = None
        if args.slice_col:
            rows = regression.residual_by_slice(df, args.slice_col)
            slices = (args.slice_col, rows)
        if args.json:
            payload = dict(summary)
            if slices:
                payload["slices"] = {"slice_column": slices[0], "rows": slices[1]}
            _print_json(payload)
        else:
            print(_regress_text(summary, slices))
        return
    cls = metrics.classification_summary(y_true, y_pred, y_proba, labels)
    err = errors.error_summary(y_true, y_pred, y_proba, labels,
                               top_n_pairs=args.top_n, top_n_hard=args.top_n)
    payload = {"classification": cls, "errors": err}
    slices = None
    if args.slice_col:
        slices = errors.slice_error_report(df, args.slice_col)
        payload["slices"] = slices
    if y_proba is not None:
        cal = calibration.calibration_summary(
            y_true, y_proba, n_bins=args.bins, labels=labels, fit_temp=not args.no_temp
        )
        payload["calibration"] = cal
    if args.json:
        _print_json(payload)
    else:
        parts = [_classify_text(cls), "", _errors_text(err, slices)]
        if y_proba is not None:
            parts += ["", _calibrate_text(payload["calibration"])]
        print("\n".join(parts))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mleval",
        description="Evaluate saved model predictions: classification, "
                    "calibration, error analysis and regression.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("input", help="prediction file (.csv or .jsonl)")
        p.add_argument("--json", action="store_true", help="emit JSON instead of text")
        return p

    p = add_common(sub.add_parser("classify", help="per-class metrics and confusion matrix"))
    p.set_defaults(func=cmd_classify)

    p = add_common(sub.add_parser("calibrate", help="reliability bins, ECE, Brier, log loss"))
    p.add_argument("--bins", type=int, default=10, help="number of probability bins (default 10)")
    p.add_argument("--no-temp", action="store_true", help="skip temperature scaling")
    p.set_defaults(func=cmd_calibrate)

    p = add_common(sub.add_parser("errors", help="confusion pairs, slices, hard examples"))
    p.add_argument("--slice-col", default=None, help="categorical column to slice errors by")
    p.add_argument("--top-n", type=int, default=10, help="how many pairs/hard examples to show")
    p.set_defaults(func=cmd_errors)

    p = add_common(sub.add_parser("report", help="full report: classify + errors + calibrate"))
    p.add_argument("--bins", type=int, default=10, help="number of probability bins (default 10)")
    p.add_argument("--no-temp", action="store_true", help="skip temperature scaling")
    p.add_argument("--slice-col", default=None, help="categorical column to slice errors by")
    p.add_argument("--top-n", type=int, default=10, help="how many pairs/hard examples to show")
    p.set_defaults(func=cmd_report)

    p = add_common(sub.add_parser("regress", help="regression metrics and residuals"))
    p.add_argument("--slice-col", default=None, help="categorical column to slice residuals by")
    p.set_defaults(func=cmd_regress)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
