# Example data

Synthetic prediction files, generated with scikit-learn's
`make_classification` / `make_regression` and a logistic-regression /
linear-regression model fitted on a training split. Everything is
seeded, so regenerating them gives identical files. There is no real
data here: no real people, no real model, no client data.

Each file has deliberate flaws, introduced after the model predicted,
so that `mleval report` shows real findings instead of a clean bill of
health. The flaws are listed here, not hidden:

## binary_predictions.csv (150 rows)

- Source: `make_classification(n_samples=500, n_features=8,
  n_informative=5, weights=[0.7, 0.3], flip_y=0.03, random_state=42)`,
  logistic regression, 70/30 train/test split (random_state=7).
- Flaw 1 — threshold 0.65 instead of 0.5. Class 1 recall drops to
  0.644 (19 of 45 positives missed).
- Flaw 2 — probabilities sharpened (p² renormalised), so the model
  looks more certain than it is. ECE is 0.091 before recalibration.
- Flaw 3 — in subgroup `south` (assigned at random, seeded), 20% of
  predictions were flipped. South error rate 0.257 vs north 0.066.

## multiclass_predictions.csv (180 rows)

- Source: `make_classification(n_samples=600, n_features=10,
  n_informative=6, n_classes=3, weights=[0.5, 0.3, 0.2], flip_y=0.03,
  random_state=11)`, logistic regression, 70/30 split (random_state=5).
- Flaw 1 — class 2 probabilities halved before argmax. Class 2 recall
  is 0.359; the top confusion pair is 2 -> 1 (18 cases).
- Flaw 2 — probabilities sharpened, as above. ECE is 0.126; the
  0.5–0.6 and 0.8–0.9 confidence bins are over-confident.
- Flaw 3 — in subgroup `beta` (assigned at random, seeded), 15% of
  predictions were flipped. Beta error rate 0.347 vs alpha 0.280.

## regression_predictions.csv (100 rows)

- Source: `make_regression(n_samples=250, n_features=5, noise=8.0,
  random_state=21)`, linear regression, 60/40 split (random_state=4).
- Flaw — subgroup `rural` predictions carry a −5 bias and extra noise
  (σ=15). Rural MAE 13.616 vs urban 6.908; rural mean residual +4.556
  (the model under-predicts rural).
- Two rows have true value exactly 0.0 (with matching zero
  predictions) so zero-safe MAPE can be checked: 2 values excluded.
