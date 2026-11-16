import numpy as np
import pandas as pd
import pytest
from src.features.email_features import EmailFeatures
from src.features.url_features import URLFeatures, URL_COLUMNS
from src.features.pipeline import run_kind, verify_kind
import json


def test_email_training_only_and_inference():
    training = pd.DataFrame({'text':['normal project meeting', 'verify account urgent!', 'project meeting tomorrow', 'verify account password'],
                             'subject':['meeting','urgent','tomorrow','verify'],'url_count':[0,1,0,1]})
    a,b = EmailFeatures(min_df=1), EmailFeatures(min_df=1)
    x,y = a.fit_transform(training), b.fit_transform(training)
    assert x.shape == y.shape and (x != y).nnz == 0
    vocabulary = dict(a.vectorizer.vocabulary_)
    scale = a.scaler.max_abs_.copy()
    new = pd.DataFrame({'text':['validationonlytoken https://evil.test/login !!!'], 'subject':['new']})
    z = a.transform(new)
    assert z.shape == (1,x.shape[1]) and np.isfinite(z.data).all()
    assert 'validationonlytoken' not in a.vectorizer.vocabulary_
    assert a.vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(scale,a.scaler.max_abs_)
    assert any(' ' in token for token in vocabulary)  # Bigrams present.
    with pytest.raises(RuntimeError): a.fit_transform(new)


def test_url_inference_columns_and_finiteness():
    transformer = URLFeatures()
    values=['https://login.example.test/a?x=1&y=2','http://192.168.1.1/verify','https://example.test/%zz']
    x = transformer.transform(values)
    assert x.shape == (3,13) and x.columns.tolist() == list(URL_COLUMNS)
    assert np.isfinite(x.to_numpy()).all()
    assert x.loc[0,'is_https']==1 and x.loc[1,'hostname_is_ip']==1
    assert x.loc[0,'query_parameter_count']==2
    pd.testing.assert_frame_equal(x,transformer.transform(values))
    assert not hasattr(transformer,'fit')


def test_pipeline_alignment_and_skip(tmp_path):
    for kind in ['emails','urls']:
        splits=tmp_path/'data/splits'/kind; splits.mkdir(parents=True)
        processed=tmp_path/'data/processed'/kind; processed.mkdir(parents=True)
        for name in ['train','validation','test']:
            if kind=='emails':
                frame=pd.DataFrame({'record_id':[name+'b',name+'a'], 'label':[1,0],
                                    'text':['verify account project','normal account project'], 'subject':['',''], 'url_count':[1,0]})
            else:
                frame=pd.DataFrame({'url_key':[name+'b',name+'a'], 'label':[1,0],
                                    'url_original':['https://login.evil.test/verify','https://normal.test/']})
            frame.to_parquet(splits/f'{name}.parquet',index=False)
        report={'split_sizes':{n:2 for n in ['train','validation','test']}}
        if kind=='urls':
            report={'validation':report}
            pd.DataFrame({'url_key':['external'], 'label':[1], 'url_original':['https://outside.test/login']}).to_parquet(processed/'phishtank_external.parquet',index=False)
        (processed/'cleaning_report.json').write_text(json.dumps(report))
        run_kind(tmp_path,kind)
        checks=verify_kind(tmp_path,kind)
        assert checks['row_alignment'] and checks['consistent_columns']
        marker=tmp_path/'data/features'/kind/'feature_report.json'
        before=marker.stat().st_mtime_ns
        run_kind(tmp_path,kind)
        assert marker.stat().st_mtime_ns==before
