# Tests: check email training behavior using saved artifacts or temporary inputs.

import json
import joblib
import numpy as np
import pandas as pd
from src.training import train_email as trainer
from src.features import email_pipeline as features
from training_checks import check_training, check_partial, check_constructors


# Test setup: build small synthetic task inputs in a temporary directory.
def make_inputs(root):
    splits = root / 'data/splits/emails'
    processed = root / 'data/processed/emails'
    splits.mkdir(parents=True)
    processed.mkdir(parents=True)
    sizes = {'train': 40, 'validation': 12, 'test': 12}
    for part, size in sizes.items():
        labels = np.arange(size) % 2
        ids = [f'{part}-{i}' for i in range(size)]
        frame = pd.DataFrame(
            dict(
                record_id=ids,
                label=labels,
                source='synthetic',
                subject='',
                url_count=labels,
                text=[
                    ('urgent verify password' if label else 'normal project meeting')
                    + ' '
                    + identity
                    for label, identity in zip(labels, ids)
                ],
                exact_hash=ids,
                duplicate_group=ids,
            )
        )
        frame.to_parquet(splits / f'{part}.parquet', index=False)
    report = {'split_sizes': sizes}
    (processed / 'cleaning_report.json').write_text(json.dumps(report))
    features.run(root)


# Test stage: independent training.
def test_independent_training(tmp_path, monkeypatch, capsys):
    check_training(
        tmp_path,
        monkeypatch,
        capsys,
        'email',
        'url',
        trainer,
        make_inputs,
        lambda p: p.predict_email('Meeting', 'normal project meeting'),
    )


# Test stage: partial artifact preservation.
def test_partial_artifact_preservation(tmp_path):
    check_partial(tmp_path, 'email', trainer, make_inputs)


# Test stage: frozen constructor equivalence.
def test_frozen_constructor_equivalence():
    check_constructors('email', trainer)


# Test stage: reproducible training.
def test_reproducible_training(tmp_path):
    make_inputs(tmp_path)
    trainer.train(tmp_path)
    x, y = trainer.load_partition(tmp_path, 'train')
    saved = joblib.load(tmp_path / 'models/email/logistic_regression_none.joblib')
    repeat = trainer.model_configs(42)['logistic_regression_none'].fit(x, y)
    np.testing.assert_allclose(saved.coef_, repeat.coef_)
