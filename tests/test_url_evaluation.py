# Tests: check url evaluation behavior using saved artifacts or temporary inputs.

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.features.url_features import URL_COLUMNS
from src.evaluation import evaluate_url as evaluation
from src.training.common import scores
from src.evaluation.common import load_selected


# Test stage: frozen artifact without fitting.
def test_frozen_artifact_without_fitting():
    root = Path(__file__).resolve().parents[1]
    model, config = load_selected(root, 'url')
    x = pd.DataFrame(np.zeros((1, len(URL_COLUMNS)), dtype=np.float32), columns=URL_COLUMNS)
    assert type(model).__name__ == 'RandomForestClassifier'
    values, score_type = scores(model, x)
    assert values.shape == (1,) and np.isfinite(values).all()
    assert score_type == config['score_type']


# Test stage: external phishing only metrics.
def test_external_phishing_only_metrics():
    result = evaluation.external_metrics(np.ones(3), np.array([0.1, 0.5, 0.9]), 0.5)
    assert result == dict(
        samples=3, phishing_recall=2 / 3, detected_phishing_urls=2, missed_phishing_urls=1
    )
    with pytest.raises(AssertionError):
        evaluation.external_metrics(np.array([0, 1]), np.array([0.1, 0.9]), 0.5)
