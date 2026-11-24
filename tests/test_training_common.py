# Tests: check training common behavior using saved artifacts or temporary inputs.

import numpy as np
import pytest
from src.training.common import metrics, tune_threshold, selection_key
from src.training.pipeline import load_partition


# Test stage: metrics and threshold.
def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1])
    values = np.array([0.1, 0.2, 0.8, 0.9])
    result = metrics(y, values, 0.5)
    assert result['average_precision'] == result['roc_auc'] == result['phishing_f1'] == 1
    assert result['confusion_matrix'] == [[2, 0], [0, 2]]
    assert metrics(y, values, tune_threshold(y, values))['phishing_f1'] == 1
    a = {
        'validation_default_threshold': dict(
            average_precision=0.9, phishing_f1=0.5, phishing_recall=0.4
        )
    }
    b = {
        'validation_default_threshold': dict(
            average_precision=0.8, phishing_f1=1.0, phishing_recall=1.0
        )
    }
    assert selection_key(a) > selection_key(b)
    with pytest.raises(ValueError):
        load_partition(None, 'email', 'test')
    with pytest.raises(ValueError):
        load_partition(None, 'url', 'external')
