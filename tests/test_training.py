import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from src.features.email_features import EmailFeatures
from src.features.url_features import URLFeatures
from src.training.pipeline import (metrics, tune_threshold, train_kind, verify_kind,
                                   model_configs, load_partition, selection_key)
import pytest


def test_metrics_and_threshold():
    y=np.array([0,0,1,1]); values=np.array([.1,.2,.8,.9])
    result=metrics(y,values,.5)
    assert result['average_precision']==result['roc_auc']==result['phishing_f1']==1
    assert result['confusion_matrix']==[[2,0],[0,2]]
    assert metrics(y,values,tune_threshold(y,values))['phishing_f1']==1
    a={'validation_default_threshold':dict(average_precision=.9,phishing_f1=.5,phishing_recall=.4)}
    b={'validation_default_threshold':dict(average_precision=.8,phishing_f1=1.,phishing_recall=1.)}
    assert selection_key(a)>selection_key(b)
    with pytest.raises(ValueError): load_partition(None,'email','test')
    with pytest.raises(ValueError): load_partition(None,'url','external')


def test_training_boundary_serialization_and_reproducibility(tmp_path,monkeypatch):
    calls=[]
    for cls in (LogisticRegression,LinearSVC,RandomForestClassifier):
        original=cls.fit
        def wrapped(self,x,y,*args,_original=original,**kwargs):
            assert len(y)==40  # Validation has 12; test/external files do not exist.
            calls.append(type(self).__name__)
            return _original(self,x,y,*args,**kwargs)
        monkeypatch.setattr(cls,'fit',wrapped)
    models=tmp_path/'models/feature_engineering'; models.mkdir(parents=True)
    for kind in ['email','url']:
        plural=kind+'s'; directory=tmp_path/'data/features'/plural; directory.mkdir(parents=True)
        splits=tmp_path/'data/splits'/plural; splits.mkdir(parents=True)
        report={'inputs':{},'validation':{'row_alignment':True,'finite_features':True}}
        transformer=EmailFeatures(min_df=1) if kind=='email' else URLFeatures()
        for part,n in [('train',40),('validation',12)]:
            y=np.arange(n)%2
            identity='record_id' if kind=='email' else 'url_key'
            if kind=='email':
                raw=pd.DataFrame({'text':['urgent verify password' if i else 'normal meeting project' for i in y],
                                  'subject':['']*n,'url_count':y,identity:[part+str(i) for i in range(n)],'label':y})
                x=transformer.fit_transform(raw) if part=='train' else transformer.transform(raw)
                sparse.save_npz(directory/f'{part}.npz',x)
                raw[[identity,'label']].to_parquet(directory/f'{part}_rows.parquet',index=False)
            else:
                raw=pd.DataFrame({'url_original':['http://login.evil.test/verify' if i else 'https://normal.test/' for i in y],
                                  identity:[part+str(i) for i in range(n)],'label':y})
                x=transformer.transform(raw.url_original)
                x[identity]=raw[identity];x['label']=y
                x.to_parquet(directory/f'{part}.parquet',index=False)
            path=splits/f'{part}.parquet';raw.to_parquet(path,index=False)
            report['inputs'][part]={'rows':n,'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
        schema=transformer.schema();report['schema']=schema
        (models/f'{plural}_schema.json').write_text(json.dumps(schema))
        (directory/'feature_report.json').write_text(json.dumps(report))
        if kind=='email':joblib.dump(transformer,models/'email_transformer.joblib')
        train_kind(tmp_path,kind,42)
        assert verify_kind(tmp_path,kind)['inference_verified']
        threshold=json.loads((tmp_path/'models'/kind/'selected_threshold.json').read_text())
        assert threshold['tuned_on']=='validation only' and np.isfinite(threshold['threshold'])
        assert (tmp_path/'models'/kind/'selected_model.joblib').is_symlink()
    assert len(calls)==6
    from src.training.pipeline import load_partition
    x,y=load_partition(tmp_path,'email','train')
    original=joblib.load(tmp_path/'models/email/logistic_regression_none.joblib')
    repeat=model_configs('email',42)['logistic_regression_none'].fit(x,y)
    np.testing.assert_allclose(original.coef_,repeat.coef_)
    scaler=joblib.load(tmp_path/'models/url/logistic_regression.joblib').named_steps['scaler']
    url_x,_=load_partition(tmp_path,'url','train')
    np.testing.assert_allclose(scaler.mean_,url_x.mean().to_numpy(),rtol=1e-5)
