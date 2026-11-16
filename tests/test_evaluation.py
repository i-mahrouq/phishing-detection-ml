from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from src.evaluation.pipeline import binary_metrics, external_metrics, finite, save_errors, load_selected
from src.training.pipeline import scores
from src.features.url_features import URL_COLUMNS
import pytest


def test_metrics_errors_and_external_restrictions(tmp_path):
    y=np.array([0,0,1,1]); values=np.array([.1,.8,.2,.9])
    m=binary_metrics(y,values,.5)
    assert m['confusion_matrix']==dict(TN=1,FP=1,FN=1,TP=1)
    assert m['accuracy']==m['phishing_f1']==m['false_positive_rate']==m['false_negative_rate']==.5
    assert m['average_precision']==pytest.approx(5/6)
    assert m['roc_auc']==.75 and finite(m)
    source=binary_metrics(np.array([0,0]),np.array([.1,.8]),.5)
    assert source['roc_auc'] is None and source['average_precision'] is None
    save_errors(tmp_path,'email_test',y,values,.5)
    errors=pd.read_parquet(tmp_path/'email_test_errors.parquet')
    assert errors.split_row.tolist()==[1,2]
    assert errors.error_type.tolist()==['false_positive','false_negative']
    assert set(errors)=={'split_row','actual_label','predicted_label','score','error_type'}
    external=external_metrics(np.ones(3),np.array([.1,.5,.9]),.5)
    assert external==dict(samples=3,phishing_recall=2/3,detected_phishing_urls=2,missed_phishing_urls=1)
    with pytest.raises(AssertionError):external_metrics(np.array([0,1]),np.array([.1,.9]),.5)


def test_frozen_artifacts_compatible_without_fitting():
    root=Path(__file__).resolve().parents[1]
    for kind in ('email','url'):
        model,config=load_selected(root,kind)
        if kind=='email':
            x=sparse.csr_matrix((1,model.n_features_in_),dtype=np.float32)
            assert type(model).__name__=='LinearSVC'
        else:
            x=pd.DataFrame(np.zeros((1,len(URL_COLUMNS)),dtype=np.float32),columns=URL_COLUMNS)
            assert type(model).__name__=='RandomForestClassifier'
        values,score_type=scores(model,x)
        assert values.shape==(1,) and np.isfinite(values).all()
        assert score_type==config['score_type']
