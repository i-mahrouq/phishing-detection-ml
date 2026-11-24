# Tests: check url training behavior using saved artifacts or temporary inputs.

import json
import joblib
import numpy as np
import pandas as pd
from src.training import train_url as trainer
from src.features import url_pipeline as features
from training_checks import check_training, check_partial, check_constructors


# Test setup: build small synthetic task inputs in a temporary directory.
def make_inputs(root):
    splits = root / 'data/splits/urls'
    processed = root / 'data/processed/urls'
    splits.mkdir(parents=True)
    processed.mkdir(parents=True)
    sizes = {'train': 40, 'validation': 12, 'test': 12}
    for part, size in sizes.items():
        labels = np.arange(size) % 2
        ids = [f'{part}-{i}' for i in range(size)]
        hosts = [f'{part}-{i}.example.test' for i in range(size)]
        frame = pd.DataFrame(
            dict(
                url_key=ids,
                label=labels,
                source='synthetic',
                hostname=hosts,
                url_original=[
                    f'http://{host}/verify?account=1' if label else f'https://{host}/'
                    for label, host in zip(labels, hosts)
                ],
            )
        )
        frame.to_parquet(splits / f'{part}.parquet', index=False)
    report = {'validation': {'split_sizes': sizes}}
    (processed / 'cleaning_report.json').write_text(json.dumps(report))
    pd.DataFrame(
        [
            dict(
                url_key='external',
                label=1,
                source='synthetic-external',
                hostname='external.test',
                url_original='http://external.test/verify',
            )
        ]
    ).to_parquet(processed / 'phishtank_external.parquet', index=False)
    features.run(root)


# Test stage: independent training.
def test_independent_training(tmp_path, monkeypatch, capsys):
    check_training(
        tmp_path,
        monkeypatch,
        capsys,
        'url',
        'email',
        trainer,
        make_inputs,
        lambda p: p.predict_url('https://normal.example.test/'),
    )


# Test stage: partial artifact preservation.
def test_partial_artifact_preservation(tmp_path):
    check_partial(tmp_path, 'url', trainer, make_inputs)


# Test stage: frozen constructor equivalence.
def test_frozen_constructor_equivalence():
    check_constructors('url', trainer)


# Test stage: reproducible training.
def test_reproducible_training(tmp_path):
    make_inputs(tmp_path)
    trainer.train(tmp_path)
    x, y = trainer.load_partition(tmp_path, 'train')
    saved = joblib.load(tmp_path / 'models/url/logistic_regression.joblib')
    np.testing.assert_allclose(saved.named_steps['scaler'].mean_, x.mean().to_numpy(), rtol=1e-5)
    repeat = trainer.model_configs(42)['logistic_regression'].fit(x, y)
    np.testing.assert_allclose(
        saved.named_steps['classifier'].coef_, repeat.named_steps['classifier'].coef_
    )
