# Training checks: verify shared fitting, preservation and model-configuration contracts.

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

from src.inference.predict import Predictor
from src.training import common
from src.evaluation import pipeline as evaluation


# Training check: verify independent fitting, saved reports and preserved inputs.
def check_training(tmp_path, monkeypatch, capsys, kind, other, module, make_inputs, predict):
    make_inputs(tmp_path)
    fits = []
    for cls in (LogisticRegression, LinearSVC, RandomForestClassifier):
        original = cls.fit

        # Test guard: record fitting inputs before calling the original fit method.
        def checked_fit(self, x, y, *args, _fit=original, **kwargs):
            assert len(y) == 40
            fits.append(type(self).__name__)
            return _fit(self, x, y, *args, **kwargs)

        monkeypatch.setattr(cls, 'fit', checked_fit)
    module.main(['--root', str(tmp_path)])
    assert len(fits) == len(module.model_configs())
    assert not (tmp_path / 'models' / other).exists()
    assert not (tmp_path / 'data/features' / (other + 's')).exists()
    assert not (tmp_path / 'reports/training/training_summary.json').exists()
    before = {str(p): p.read_bytes() for p in (tmp_path / 'models').rglob('*') if p.is_file()}

    # Test guard: fail if a forbidden training or data operation is called.
    def forbidden(*args, **kwargs):
        raise AssertionError('Comparison/evaluation/inference must not fit or tune')

    for cls in (LogisticRegression, LinearSVC, RandomForestClassifier):
        monkeypatch.setattr(cls, 'fit', forbidden)
    monkeypatch.setattr(common, 'tune_threshold', forbidden)
    comparison = module.compare(tmp_path)
    assert comparison['partition'] == 'validation' and not comparison['test_evaluated']
    for record in comparison['configurations']:
        assert (
            set(
                [
                    'accuracy',
                    'phishing_precision',
                    'phishing_recall',
                    'phishing_f1',
                    'confusion_matrix',
                ]
            )
            <= record['metrics'].keys()
        )
        assert record['training_seconds'] >= 0 and record['validation_inference_seconds'] >= 0

    predictor = Predictor(tmp_path)
    result = predict(predictor)
    assert result['prediction'] in ('PHISHING', 'LEGITIMATE')
    output_directory = tmp_path / 'reports/evaluation'
    output_directory.mkdir()
    orphan = output_directory / f'{kind}_confusion_matrix.png'
    orphan.write_bytes(b'preserved evidence')
    with pytest.raises(RuntimeError, match='refusing to overwrite'):
        evaluation.run(tmp_path, kind)
    assert orphan.read_bytes() == b'preserved evidence'
    assert not (output_directory / f'{kind}_evaluation_started.json').exists()
    orphan.unlink()
    evaluation.run(tmp_path, kind)
    assert evaluation.verify(tmp_path, kind)['reports_verified']
    assert not (tmp_path / 'reports/evaluation' / f'{other}_test_metrics.json').exists()
    assert before == {
        str(p): p.read_bytes() for p in (tmp_path / 'models').rglob('*') if p.is_file()
    }

    entry = f'src.training.train_{kind}'
    for mode in ('--verify', '--compare'):
        command = subprocess.run(
            [sys.executable, '-m', entry, '--root', str(tmp_path), mode],
            capture_output=True,
            text=True,
        )
        assert command.returncode == 0, command.stderr
        assert json.loads(command.stdout)
    output = tmp_path / 'comparison.json'
    module.main(['--root', str(tmp_path), '--compare', '--output', str(output)])
    saved = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        module.main(['--root', str(tmp_path), '--compare', '--output', str(output)])
    assert error.value.code == 2 and output.read_bytes() == saved
    capsys.readouterr()


# Preservation check: refuse to overwrite artifacts from incomplete runs.
def check_partial(tmp_path, kind, module, make_inputs):
    make_inputs(tmp_path)
    directory = tmp_path / 'models' / kind
    directory.mkdir()
    artifact = directory / 'selected_threshold.json'
    artifact.write_text('{"preserve": true}')
    with pytest.raises(SystemExit) as error:
        module.main(['--root', str(tmp_path)])
    assert error.value.code == 2 and artifact.read_text() == '{"preserve": true}'
    assert not (directory / 'training_started.json').exists()


# Configuration check: compare retained model parameters with the frozen definitions.
def check_constructors(kind, module):
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / 'reports/training' / f'{kind}_model_comparison.json').read_text())
    configurations = module.model_configs(42)
    assert list(configurations) == [record['name'] for record in report['configurations']]
    for record in report['configurations']:
        fresh = configurations[record['name']]
        saved = joblib.load(root / record['artifact'])
        assert type(fresh) is type(saved)
        if hasattr(saved, 'named_steps'):
            assert list(fresh.named_steps) == list(saved.named_steps)
            for name in saved.named_steps:
                assert fresh.named_steps[name].get_params() == saved.named_steps[name].get_params()
        else:
            assert fresh.get_params() == saved.get_params()
