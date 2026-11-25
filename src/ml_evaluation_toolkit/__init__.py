"""ml-evaluation-toolkit: evaluate saved model predictions.

Classification metrics, error analysis, calibration and regression
evaluation. This library evaluates predictions; it does not train
models.
"""

from . import calibration, errors, metrics, regression

__all__ = ["calibration", "errors", "metrics", "regression"]
__version__ = "0.1.0"
