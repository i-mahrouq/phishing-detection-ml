import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from src.inference.predict import Predictor, email_frame, main, risk_level
from src.preprocessing.emails import clean, urls


def test_risk_rules_and_input_validation():
    for kind,width in [('email',.5),('url',.1)]:
        assert risk_level(kind,-width-.01,0)=='LOW'
        assert risk_level(kind,-width,0)=='MEDIUM'
        assert risk_level(kind,0,0)=='MEDIUM'
        assert risk_level(kind,width,0)=='HIGH'
    for subject,body in [('', ''),('', '  '),('', '<html><script>alert(1)</script></html>'),('',None)]:
        with pytest.raises(ValueError):email_frame(subject,body)
    predictor=Predictor()
    for url in ['', 'javascript:alert(1)', 'https://bad host.test', 'https://example.test:99999']:
        with pytest.raises(ValueError):predictor.predict_url(url)


def test_frozen_inference_and_feature_parity(monkeypatch):
    def forbidden(*a,**k): raise AssertionError('Inference must never fit')
    monkeypatch.setattr(TfidfVectorizer,'fit',forbidden)
    monkeypatch.setattr(TfidfVectorizer,'fit_transform',forbidden)
    monkeypatch.setattr(MaxAbsScaler,'fit',forbidden)
    predictor=Predictor()
    paths=[predictor.root/'models'/kind/name for kind in ['email','url']
           for name in ['selected_model.joblib','selected_threshold.json']]
    paths+=[predictor.root/'models/feature_engineering/email_transformer.joblib']
    before=[p.stat().st_mtime_ns for p in paths]
    subject='Urgent verification'
    body='<p>Verify account!!! <a href="https://example.test/login">click</a></p>'
    frame=email_frame(subject,body)
    expected=pd.DataFrame([{'text':clean(subject+'\n'+body),'subject':clean(subject),
                            'url_count':len(urls(subject+'\n'+body))}])
    pd.testing.assert_frame_equal(frame,expected)
    assert frame.url_count.iloc[0]==1
    for kind,result in [('email',predictor.predict_email(subject,body)),
                        ('url',predictor.predict_url('https://example.test/login?account=1'))]:
        model,transformer,config=predictor._loaded[kind]
        assert np.isfinite(result['score']) and result['threshold']==config['threshold']
        assert result['prediction']==('PHISHING' if result['score']>=config['threshold'] else 'LEGITIMATE')
        assert result['risk_level'] in ['LOW','MEDIUM','HIGH']
        assert predictor._load(kind)[0] is model
        if kind=='email':
            x=transformer.transform(expected)
            assert result['score']==float(model.decision_function(x)[0])
            assert result['score_type']=='uncalibrated_margin'
    assert before==[p.stat().st_mtime_ns for p in paths]


def test_cli_json_errors_and_interactive(capsys,monkeypatch):
    main(['email','--subject','Meeting','--body','Project meeting tomorrow','--json'])
    result=json.loads(capsys.readouterr().out)
    assert result['score_type']=='uncalibrated_margin' and 'probability' not in result
    main(['url','--url','https://example.test/','--json'])
    assert json.loads(capsys.readouterr().out)['score_type']=='probability'
    with pytest.raises(SystemExit) as error:main(['url','--url','bad'])
    assert error.value.code==2
    assert 'Error:' in capsys.readouterr().err
    answers=iter(['email','Meeting','Project meeting tomorrow','.','url','https://example.test/','quit'])
    monkeypatch.setattr('builtins.input',lambda *a:next(answers))
    main(['--interactive'])
    assert capsys.readouterr().out.count('Prediction:')==2
