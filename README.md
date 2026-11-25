# ml-evaluation-toolkit

Evaluate saved model predictions: classification metrics, error
analysis, calibration and regression evaluation. It evaluates
predictions a model has already made — it does not train models.

Built as public proof of evaluation work: the metrics are computed
from first principles where that is practical (confusion matrix,
per-class precision/recall/F1, MCC, ECE, Brier, MAPE), and delegated
to scikit-learn where reimplementing would only add risk (ROC-AUC,
average precision). The test suite cross-checks the hand-computed
metrics against scikit-learn to a tight tolerance.

## Install

```bash
pip install -e .
```

Requires Python 3.10+, NumPy, pandas and scikit-learn (declared in
`pyproject.toml`). Tests need pytest.

## Usage

Prediction files are CSV or JSONL with `y_true` and `y_pred` columns,
plus optional probabilities as a single `y_proba` column (binary
positive-class probability) or `proba_0`, `proba_1`, … columns. Any
extra columns are kept — they are what slicing runs on.

```bash
mleval classify data/examples/binary_predictions.csv
mleval calibrate data/examples/binary_predictions.csv --bins 10
mleval errors data/examples/binary_predictions.csv --slice-col group
mleval regress data/examples/regression_predictions.csv --slice-col group
mleval report data/examples/multiclass_predictions.csv --slice-col group
```

Every command takes `--json` for machine-readable output. `report`
runs classify + errors + calibrate together, and auto-detects
regression files (float targets) and reports those instead.

## What it measures

**Classification** (`ml_evaluation_toolkit.metrics`): per-class
precision/recall/F1/support, macro and weighted averages, confusion
matrix (counts and row-normalised), accuracy, balanced accuracy,
Matthews correlation coefficient, ROC-AUC and average precision
(binary, or macro one-vs-rest for multiclass).

**Error analysis** (`errors`): confusion pairs with example row
indices, error rate by true and by predicted class, error rate broken
down by any categorical feature column, and a "hard examples" list —
the misclassified rows with the highest loss (−log P(true class)).

**Calibration** (`calibration`): reliability-diagram bins (mean
predicted vs observed frequency, with an over/under/well-calibrated
verdict per bin), Expected Calibration Error, Brier score (binary and
multiclass), log loss, and temperature scaling — a single temperature
fitted by golden-section search on negative log-likelihood, returned
with the recalibrated probabilities so the improvement can be checked
on held-out data.

**Regression** (`regression`): MAE, RMSE, R², and MAPE with a
zero-safe definition (rows with true value 0 are excluded and the
count of excluded rows is reported, not hidden), plus residual
statistics by feature slice.

## Sample output

Real output from `mleval report data/examples/binary_predictions.csv
--slice-col group` (see `data/examples/README.md` for how this file
was made and which flaws were deliberately built into it):

```
Classification report
=====================
n = 150
accuracy          0.840
balanced accuracy 0.784
macro F1          0.799
weighted F1       0.835
MCC               0.604
ROC-AUC (macro)   0.880
avg precision     0.864

       label  precision   recall       f1  support
           0      0.858    0.924    0.890      105
           1      0.784    0.644    0.707       45

Confusion matrix (rows = true, columns = predicted)
                   0       1
           0      97       8
           1      16      29

Error analysis
==============
total 150  errors 24  error rate 0.160

Top confusion pairs (true -> predicted)
  1 -> 0: 16 (examples [8, 9, 20, 24, 30])
  0 -> 1: 8 (examples [17, 52, 64, 71, 88])

Error rate by true class
  0: 0.076 (8/105)
  1: 0.356 (16/45)

Most confidently wrong examples (highest loss first)
  row 8: true=1 pred=0 P(true)=0.000 loss=13.816
  row 56: true=1 pred=0 P(true)=0.000 loss=10.982
  row 101: true=1 pred=0 P(true)=0.001 loss=7.245
  row 30: true=1 pred=0 P(true)=0.007 loss=4.952
  row 9: true=1 pred=0 P(true)=0.019 loss=3.989
  row 95: true=1 pred=0 P(true)=0.022 loss=3.835
  row 47: true=1 pred=0 P(true)=0.034 loss=3.373
  row 24: true=1 pred=0 P(true)=0.037 loss=3.306
  row 32: true=1 pred=0 P(true)=0.053 loss=2.941
  row 96: true=0 pred=1 P(true)=0.122 loss=2.100

Error rate by group (overall 0.160)
  south: 0.257 (19/74)
  north: 0.066 (5/76)

Calibration report
==================
ECE        0.091
Brier      0.099
log loss   0.489
temperature (fitted) 2.317
calibrated ECE        0.046
calibrated Brier      0.097
calibrated log loss   0.358

           bin     n  mean pred  observed     gap  verdict
    [0.0, 0.1)    93      0.013     0.097  -0.084  under-confident
    [0.1, 0.2)     9      0.157     0.000   0.157  over-confident
    [0.2, 0.3)     9      0.246     0.444  -0.199  under-confident
    [0.3, 0.4)     3      0.342     0.333   0.009  well calibrated
    [0.5, 0.6)     4      0.564     0.250   0.314  over-confident
    [0.7, 0.8)     2      0.762     0.500   0.262  over-confident
    [0.8, 0.9)     2      0.841     0.500   0.341  over-confident
    [0.9, 1.0]    28      0.993     1.000  -0.007  well calibrated
```

The findings in that run are the deliberate flaws in the example:
class 1 recall is only 0.644 (a 0.65 decision threshold), the
`south` subgroup errs almost four times as often as `north`, and the
probabilities are over-confident — temperature scaling (T = 2.317)
halves the ECE from 0.091 to 0.046 and cuts log loss from 0.489 to
0.358.

The multiclass example (`multiclass_predictions.csv`) shows the same
machinery on three classes: class 2 recall 0.359 with the top
confusion pair 2 -> 1 (18 cases), accuracy 0.683, ECE 0.126 falling
to 0.037 after temperature scaling (T = 1.911). The regression
example shows a subgroup residual gap:

```
Regression report
=================
n    100
MAE  10.396
RMSE 14.143
R2   0.989
MAPE 0.693 (2 zero true values excluded)

Residuals by group
  rural: MAE 13.616 RMSE 17.828 mean residual 4.556 (n=52)
  urban: MAE 6.908 RMSE 8.508 mean residual -1.002 (n=48)
```

## Design decisions, stated honestly

- **Temperature scaling, not Platt/isotonic.** Recalibration is a
  single temperature fitted on log-likelihood by golden-section
  search. It repairs the *scale* of miscalibrated probabilities; it
  cannot repair a model that ranks examples badly. Isotonic
  regression was left out deliberately: on small calibration sets it
  overfits, and a one-parameter fit is the honest default. Fit the
  temperature on a calibration split and evaluate on held-out data —
  fitting and measuring on the same rows flatters the calibrated
  numbers.
- **ECE is binned, and bins are fragile.** With 10 equal-width bins,
  a bin with 2 rows says almost nothing; the report shows the count
  next to every bin so a thin bin is visible. Temperature scaling
  optimises log loss, not ECE, so a calibrated ECE can occasionally
  move the wrong way while log loss improves — that is expected, not
  a bug.
- **Subgroup slices are only as good as the column supplied.** A
  slice column that leaks the label will flatter the model, and a
  column with one value per row says nothing. The tool reports the
  slice; judging whether the slice is meaningful is the reviewer's
  job.
- **This evaluates predictions, it does not train models.** There is
  no model fitting anywhere in the library. Calibration metrics
  describe the probabilities you hand it; they do not fix a bad
  model.

## Tests

```bash
python -m pytest -q
```

93 tests: metrics cross-checked against scikit-learn
(`precision_recall_fscore_support`, `accuracy_score`,
`balanced_accuracy_score`, `confusion_matrix`, `matthews_corrcoef`,
`roc_auc_score`, `average_precision_score`, `brier_score_loss`,
`log_loss`, `mean_absolute_error`, `mean_squared_error`, `r2_score`,
`mean_absolute_percentage_error`), calibration on hand-computed
cases, error analysis on known inputs, and CLI end-to-end runs on
the example data asserting the real numbers.

## License

MIT.
