# Tests: check email evaluation behavior using saved artifacts or temporary inputs.

from pathlib import Path
import numpy as np
from scipy import sparse
from src.evaluation import evaluate_email as evaluation
from src.training.common import scores
from src.evaluation.common import load_selected


# Test stage: frozen artifact without fitting.
def test_frozen_artifact_without_fitting():
    root = Path(__file__).resolve().parents[1]
    model, config = load_selected(root, 'email')
    x = sparse.csr_matrix((1, model.n_features_in_), dtype=np.float32)
    assert type(model).__name__ == 'LinearSVC'
    values, score_type = scores(model, x)
    assert values.shape == (1,) and np.isfinite(values).all()
    assert score_type == config['score_type']
