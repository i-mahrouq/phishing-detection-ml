"""Independent task entry points and validation/test boundaries on temporary data."""
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src.features.pipeline import run_kind
from src.inference.predict import Predictor
from src.training import common, train_email, train_url
from src.evaluation import pipeline as evaluation


def make_task_inputs(root, kind):
    """Only this task exists; held-out rows have distinct provenance identities."""
    plural = kind + 's'
    splits = root/'data/splits'/plural
    processed = root/'data/processed'/plural
    splits.mkdir(parents=True)
    processed.mkdir(parents=True)
    sizes = {'train': 40, 'validation': 12, 'test': 12}
    for part, size in sizes.items():
        labels = np.arange(size) % 2
        ids = [f'{part}-{i}' for i in range(size)]
        if kind == 'email':
            frame = pd.DataFrame(dict(
                record_id=ids, label=labels, source='synthetic', subject='', url_count=labels,
                text=[('urgent verify password' if label else 'normal project meeting') + ' ' + identity
                      for label, identity in zip(labels, ids)],
                exact_hash=ids, duplicate_group=ids))
        else:
            hosts = [f'{part}-{i}.example.test' for i in range(size)]
            frame = pd.DataFrame(dict(
                url_key=ids, label=labels, source='synthetic', hostname=hosts,
                url_original=[f'http://{host}/verify?account=1' if label else f'https://{host}/'
                              for label, host in zip(labels, hosts)]))
        frame.to_parquet(splits/f'{part}.parquet', index=False)
    report = {'split_sizes': sizes} if kind == 'email' else {'validation': {'split_sizes': sizes}}
    (processed/'cleaning_report.json').write_text(json.dumps(report))
    if kind == 'url':
        pd.DataFrame([dict(url_key='external', label=1, source='synthetic-external',
                           hostname='external.test', url_original='http://external.test/verify')]
                     ).to_parquet(processed/'phishtank_external.parquet', index=False)
    run_kind(root, plural)


@pytest.mark.parametrize('kind,module', [('email', train_email), ('url', train_url)])
def test_independent_training_comparison_and_selected_test(tmp_path, monkeypatch, capsys, kind, module):
    make_task_inputs(tmp_path, kind)
    other = 'url' if kind == 'email' else 'email'
    fits = []
    for cls in (LogisticRegression, LinearSVC, RandomForestClassifier):
        original = cls.fit

        def checked_fit(self, x, y, *args, _fit=original, **kwargs):
            assert len(y) == 40  # Validation/test have 12; never fit those.
            fits.append(type(self).__name__)
            return _fit(self, x, y, *args, **kwargs)

        monkeypatch.setattr(cls, 'fit', checked_fit)
    module.main(['--root', str(tmp_path)])
    assert len(fits) == (4 if kind == 'email' else 2)
    assert not (tmp_path/'models'/other).exists()
    assert not (tmp_path/'data/features'/(other+'s')).exists()
    assert not (tmp_path/'reports/training/training_summary.json').exists()
    before = {str(p): p.read_bytes() for p in (tmp_path/'models').rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError('Comparison/evaluation/inference must not fit or tune')

    for cls in (LogisticRegression, LinearSVC, RandomForestClassifier):
        monkeypatch.setattr(cls, 'fit', forbidden)
    monkeypatch.setattr(common, 'tune_threshold', forbidden)
    comparison = common.compare_kind(tmp_path, kind)
    assert comparison['partition'] == 'validation' and not comparison['test_evaluated']
    for record in comparison['configurations']:
        assert set(['accuracy', 'phishing_precision', 'phishing_recall', 'phishing_f1',
                    'confusion_matrix']) <= record['metrics'].keys()
        assert record['training_seconds'] >= 0 and record['validation_inference_seconds'] >= 0

    # A temporary selection may choose LR: every retained family must load.
    predictor = Predictor(tmp_path)
    result = (predictor.predict_email('Meeting', 'normal project meeting') if kind == 'email'
              else predictor.predict_url('https://normal.example.test/'))
    assert result['prediction'] in ('PHISHING', 'LEGITIMATE')
    output_directory = tmp_path/'reports/evaluation'
    output_directory.mkdir()
    orphan = output_directory/f'{kind}_confusion_matrix.png'
    orphan.write_bytes(b'preserved evidence')
    with pytest.raises(RuntimeError, match='refusing to overwrite'):
        evaluation.run(tmp_path, kind)
    assert orphan.read_bytes() == b'preserved evidence'
    assert not (output_directory/f'{kind}_evaluation_started.json').exists()
    orphan.unlink()  # Only the synthetic file created by this test.
    evaluation.run(tmp_path, kind)
    assert evaluation.verify(tmp_path, kind)['reports_verified']
    assert not (tmp_path/'reports/evaluation'/f'{other}_test_metrics.json').exists()
    assert before == {str(p): p.read_bytes() for p in (tmp_path/'models').rglob('*') if p.is_file()}

    entry = f'src.training.train_{kind}'
    for mode in ('--verify', '--compare'):
        command = subprocess.run([sys.executable, '-m', entry, '--root', str(tmp_path), mode],
                                 capture_output=True, text=True)
        assert command.returncode == 0, command.stderr
        assert json.loads(command.stdout)
    output = tmp_path/'comparison.json'
    module.main(['--root', str(tmp_path), '--compare', '--output', str(output)])
    saved = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        module.main(['--root', str(tmp_path), '--compare', '--output', str(output)])
    assert error.value.code == 2 and output.read_bytes() == saved
    capsys.readouterr()


@pytest.mark.parametrize('kind,module', [('email', train_email), ('url', train_url)])
def test_partial_model_artifacts_are_preserved(tmp_path, kind, module):
    make_task_inputs(tmp_path, kind)
    directory = tmp_path/'models'/kind
    directory.mkdir()
    artifact = directory/'selected_threshold.json'
    artifact.write_text('{"preserve": true}')
    with pytest.raises(SystemExit) as error:
        module.main(['--root', str(tmp_path)])
    assert error.value.code == 2 and artifact.read_text() == '{"preserve": true}'
    assert not (directory/'training_started.json').exists()


@pytest.mark.parametrize('kind,module', [('email', train_email), ('url', train_url)])
def test_task_constructors_match_frozen_v1(kind, module):
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/'reports/training'/f'{kind}_model_comparison.json').read_text())
    configurations = module.model_configs(42)
    assert list(configurations) == [record['name'] for record in report['configurations']]
    for record in report['configurations']:
        fresh = configurations[record['name']]
        saved = joblib.load(root/record['artifact'])
        assert type(fresh) is type(saved)
        if hasattr(saved, 'named_steps'):
            assert list(fresh.named_steps) == list(saved.named_steps)
            for name in saved.named_steps:
                assert fresh.named_steps[name].get_params() == saved.named_steps[name].get_params()
        else:
            assert fresh.get_params() == saved.get_params()
