"""Final evaluation only: no fit, feature transformation, or threshold selection."""
import argparse
import json
import logging
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from src.training.pipeline import scores, write_json
from src.features.url_features import URL_COLUMNS


def binary_metrics(y, values, threshold):
    predicted = (values >= threshold).astype(np.int8)
    tn,fp,fn,tp = confusion_matrix(y,predicted,labels=[0,1]).ravel().tolist()
    both = len(np.unique(y)) == 2
    return dict(samples=len(y), accuracy=float(accuracy_score(y,predicted)),
                phishing_precision=float(precision_score(y,predicted,zero_division=0)),
                phishing_recall=float(recall_score(y,predicted,zero_division=0)),
                phishing_f1=float(f1_score(y,predicted,zero_division=0)),
                average_precision=float(average_precision_score(y,values)) if both else None,
                roc_auc=float(roc_auc_score(y,values)) if both else None,
                confusion_matrix=dict(TN=tn,FP=fp,FN=fn,TP=tp),
                false_positive_rate=fp/(fp+tn) if fp+tn else None,
                false_negative_rate=fn/(fn+tp) if fn+tp else None,
                ranking_metrics_defined=both)


def external_metrics(y, values, threshold):
    assert len(y)>0 and np.all(y == 1), 'External dataset must be phishing-only'
    detected=int(np.count_nonzero(values >= threshold))
    return dict(samples=len(y),phishing_recall=detected/len(y),
                detected_phishing_urls=detected,missed_phishing_urls=len(y)-detected)


def finite(value):
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,list):
        return all(finite(v) for v in value)
    return bool(np.isfinite(value)) if isinstance(value,(int,float)) else True


def frozen_paths(root):
    paths=[]
    for kind in ('email','url'):
        paths += [root/'models'/kind/'selected_model.joblib',root/'models'/kind/'selected_threshold.json',
                  root/'reports/training'/f'{kind}_model_comparison.json',
                  root/'models/feature_engineering'/f'{kind}s_schema.json',
                  root/'data/features'/(kind+'s')/'feature_report.json']
        paths += [root/'data/features'/(kind+'s')/('test.npz' if kind=='email' else 'test.parquet')]
    paths += [root/'models/feature_engineering/email_transformer.joblib',
              root/'data/features/emails/test_rows.parquet',root/'data/features/urls/external.parquet']
    return {str(p.relative_to(root)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}


def load_data(root, kind, part='test'):
    plural=kind+'s'; directory=root/'data/features'/plural
    identity='record_id' if kind=='email' else 'url_key'
    report=json.loads((directory/'feature_report.json').read_text())
    schema=json.loads((root/'models/feature_engineering'/f'{plural}_schema.json').read_text())
    assert schema==report['schema']
    original=root/('data/processed/urls/phishtank_external.parquet' if part=='external' else f'data/splits/{plural}/test.parquet')
    metadata=report['inputs'][part]
    assert original.stat().st_size==metadata['size'] and original.stat().st_mtime_ns==metadata['mtime_ns']
    columns=[identity,'label','source']
    if kind=='email': columns+=['exact_hash','duplicate_group']
    else: columns+=['hostname']
    original_rows=pd.read_parquet(original,columns=columns)
    if kind=='email':
        assert part=='test'
        x=sparse.load_npz(directory/'test.npz')
        rows=pd.read_parquet(directory/'test_rows.parquet')
        assert sparse.isspmatrix_csr(x) and np.isfinite(x.data).all()
        assert x.shape[1]==schema['n_features']
    else:
        frame=pd.read_parquet(directory/f'{part}.parquet')
        assert frame.columns.tolist()==list(URL_COLUMNS)+[identity,'label']
        x=frame[list(URL_COLUMNS)]; rows=frame[[identity,'label']]
        assert np.isfinite(x.to_numpy()).all()
    pd.testing.assert_frame_equal(rows.reset_index(drop=True),original_rows[[identity,'label']].reset_index(drop=True))
    assert x.shape[0]==len(rows)==metadata['rows'] and rows[identity].is_unique
    y=rows.label.to_numpy()
    assert set(np.unique(y))==({1} if part=='external' else {0,1})
    return x,y,original_rows


def load_selected(root, kind):
    directory=root/'models'/kind
    config=json.loads((directory/'selected_threshold.json').read_text())
    comparison=json.loads((root/'reports/training'/f'{kind}_model_comparison.json').read_text())
    assert config['model']==comparison['selected_model']
    assert config['threshold']==comparison['selected_threshold']
    assert (directory/'selected_model.joblib').resolve()==(root/config['artifact']).resolve()
    assert config['positive_label']==1 and config['prediction_rule']=='score >= threshold'
    assert np.isfinite(config['threshold'])
    model=joblib.load(directory/'selected_model.joblib')
    assert list(model.classes_)==[0,1]
    expected='uncalibrated_margin' if kind=='email' else 'probability'
    assert config['score_type']==expected
    assert hasattr(model,'decision_function') if kind=='email' else hasattr(model,'predict_proba')
    return model,config


def save_errors(out, name, y, values, threshold):
    predicted=(values>=threshold).astype(np.int8)
    bad=predicted!=y
    # Split row positions identify errors locally without exposing content/URLs.
    pd.DataFrame(dict(split_row=np.flatnonzero(bad),actual_label=y[bad],predicted_label=predicted[bad],
                      score=values[bad],error_type=np.where(y[bad]==0,'false_positive','false_negative'))).to_parquet(out/f'{name}_errors.parquet',index=False)


def plot_matrix(out, name, matrix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    values=np.array([[matrix['TN'],matrix['FP']],[matrix['FN'],matrix['TP']]])
    figure,axis=plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay(values,display_labels=['Legitimate','Phishing']).plot(ax=axis,cmap='Blues',values_format='d',colorbar=False)
    axis.set_title(f'{name.capitalize()} test — saved threshold')
    figure.tight_layout(); figure.savefig(out/f'{name}_confusion_matrix.png',dpi=150);plt.close(figure)


def overlap(root,kind,rows):
    columns=['exact_hash','duplicate_group'] if kind=='email' else ['url_key','hostname']
    train=pd.read_parquet(root/'data/splits'/(kind+'s')/'train.parquet',columns=columns)
    return {f'train_test_{column}_overlap':len(set(train[column]) & set(rows[column])) for column in columns}


def verify(root):
    out=root/'reports/evaluation'
    summary=json.loads((out/'evaluation_summary.json').read_text())
    assert summary['completed'] and summary['frozen_artifacts']==frozen_paths(root)
    for name in ('email_test','url_test','phishtank_external'):
        report=json.loads((out/f'{name}_metrics.json').read_text())
        assert finite(report)
        kind='email' if name=='email_test' else 'url'
        config=json.loads((root/'models'/kind/'selected_threshold.json').read_text())
        assert report['threshold']==config['threshold'] and report['selected_model']==config['model']
        if name=='phishtank_external':
            assert report['detected_phishing_urls']+report['missed_phishing_urls']==report['samples']
            assert not {'accuracy','phishing_precision','roc_auc','average_precision','false_positive_rate'} & report.keys()
        else:
            assert sum(report['confusion_matrix'].values())==report['samples']
            assert (out/f'{kind}_confusion_matrix.png').is_file()
            errors=pd.read_parquet(out/f'{kind}_test_errors.parquet',columns=['error_type'])
            assert (errors.error_type=='false_positive').sum()==report['confusion_matrix']['FP']
            assert (errors.error_type=='false_negative').sum()==report['confusion_matrix']['FN']
    return dict(completed=True,models_and_thresholds_unchanged=True,reports_verified=True)


def run(root):
    out=root/'reports/evaluation'
    if (out/'evaluation_summary.json').exists():
        logging.info('Existing evaluation verified: %s',verify(root));return
    assert json.loads((root/'reports/training/training_summary.json').read_text())['completed']
    before=frozen_paths(root)
    out.mkdir(parents=True,exist_ok=True)
    marker=out/'evaluation_started.json'
    if marker.exists(): raise RuntimeError('Partial prior evaluation detected; inspect logs rather than automatically rerun')
    write_json(marker,{'frozen_artifacts':before})
    flags=[];results={}
    for kind in ('email','url'):
        logging.info('Evaluating %s test',kind)
        model,config=load_selected(root,kind)
        x,y,rows=load_data(root,kind)
        assert model.n_features_in_==x.shape[1]
        started=time.perf_counter();values,score_type=scores(model,x);elapsed=time.perf_counter()-started
        assert score_type==config['score_type']
        result=binary_metrics(y,values,config['threshold'])
        result.update(selected_model=config['model'],threshold=config['threshold'],score_type=score_type,
                      inference_seconds=elapsed,inference_ms_per_sample=1000*elapsed/len(y),
                      timing_scope='model scoring only; excludes loading and feature extraction',
                      overlap_checks=overlap(root,kind,rows),label_mapping={'0':'legitimate','1':'phishing'})
        result['by_source']={str(source):binary_metrics(y[indices],values[indices],config['threshold'])
                             for source,indices in rows.groupby('source').indices.items()}
        result['limitations']=json.loads((root/'data/features'/(kind+'s')/'feature_report.json').read_text()).get('input_limitations',[])
        if any(result['overlap_checks'].values()):flags.append(f'{kind}: detected train/test URL, body, host or group overlap; investigate')
        if result['average_precision']>=.99 or result['roc_auc']>=.99:
            flags.append(f'{kind}: very high ranking metrics warrant review of corpus artifacts, source bias and leakage; not evidence of leakage alone')
        assert finite(result)
        save_errors(out,f'{kind}_test',y,values,config['threshold'])
        plot_matrix(out,kind,result['confusion_matrix'])
        write_json(out/f'{kind}_test_metrics.json',result);results[kind]=result
        del x,values,rows
        if kind=='url':
            logging.info('Evaluating phishing-only PhishTank external data')
            x,y,rows=load_data(root,'url','external')
            assert model.n_features_in_==x.shape[1]
            values,score_type=scores(model,x)
            external=external_metrics(y,values,config['threshold'])
            external.update(selected_model=config['model'],threshold=config['threshold'],score_type=score_type)
            checks={}
            for part in ('train','validation','test'):
                primary=pd.read_parquet(root/'data/splits/urls'/f'{part}.parquet',columns=['url_key','hostname'])
                for column in ('url_key','hostname'):
                    checks[f'{part}_external_{column}_overlap']=len(set(primary[column]) & set(rows[column]))
            external['overlap_checks']=checks
            if any(checks.values()): flags.append('PhishTank overlaps primary URL datasets; review external independence')
            write_json(out/'phishtank_external_metrics.json',external)
            results['external']=external
            del x,y,values,rows
    assert before==frozen_paths(root),'Frozen inputs changed during evaluation'
    write_json(out/'evaluation_summary.json',dict(completed=True,frozen_artifacts=before,
               report_files=['email_test_metrics.json','url_test_metrics.json','phishtank_external_metrics.json'],
               review_flags=flags,models_changed=False,thresholds_changed=False,
               source_analysis_note='Single-class source groups have undefined ranking metrics and rates set to null.',
               privacy='Error artifacts contain split row positions, labels and scores only; no raw content or URLs.',
               results={name:{k:v for k,v in result.items() if k in ('samples','accuracy','phishing_recall','phishing_f1','average_precision')} for name,result in results.items()}))
    logging.info('Evaluation verified: %s',verify(root))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    with threadpool_limits(limits=2):
        if args.verify: print(json.dumps(verify(args.root)))
        else: run(args.root)

if __name__=='__main__':main()
