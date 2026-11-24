# Tests: check evaluation common behavior using saved artifacts or temporary inputs.

import numpy as np
import pandas as pd
import pytest
from src.evaluation.common import binary_metrics, finite, save_errors


# Test stage: metrics and error records.
def test_metrics_and_error_records(tmp_path):
    y = np.array([0, 0, 1, 1])
    values = np.array([0.1, 0.8, 0.2, 0.9])
    m = binary_metrics(y, values, 0.5)
    assert m['confusion_matrix'] == dict(TN=1, FP=1, FN=1, TP=1)
    assert (
        m['accuracy']
        == m['phishing_f1']
        == m['false_positive_rate']
        == m['false_negative_rate']
        == 0.5
    )
    assert m['average_precision'] == pytest.approx(5 / 6)
    assert m['roc_auc'] == 0.75 and finite(m)
    source = binary_metrics(np.array([0, 0]), np.array([0.1, 0.8]), 0.5)
    assert source['roc_auc'] is None and source['average_precision'] is None
    save_errors(tmp_path, 'email_test', y, values, 0.5)
    errors = pd.read_parquet(tmp_path / 'email_test_errors.parquet')
    assert errors.split_row.tolist() == [1, 2]
    assert errors.error_type.tolist() == ['false_positive', 'false_negative']
    assert set(errors) == {'split_row', 'actual_label', 'predicted_label', 'score', 'error_type'}
