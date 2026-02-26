import pandas as pd
from skeval import Model
import pytest


def test_model_metrics():
    # Confusion matrix: [[tn, fp], [fn, tp]]
    matrix = pd.DataFrame([[50, 10], [5, 35]])

    m = Model(matrix, model="TestModel")
    m.get_results()

    assert m.results['sensitivity'] == pytest.approx(35 / (35 + 5))
    assert m.results['specificity'] == pytest.approx(50 / (50 + 10))
    assert m.results['accuracy'] == pytest.approx((35 + 50) / 100)
    assert m.results['error_rate'] == pytest.approx((10 + 5) / 100)
