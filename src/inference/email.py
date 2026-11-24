# Email inference: prepare subject and body, load saved artifacts and predict with the fixed threshold.

import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd
from src.preprocessing.emails import clean, urls
from .common import DEFAULT_ROOT, metadata, read_artifact, validate_model, result, display

RISK_WIDTH = 0.5


# Preparation stage: clean subject and body and count links before removing HTML.
def email_frame(subject: str, body: str) -> pd.DataFrame:
    if not isinstance(subject, str) or not isinstance(body, str):
        raise ValueError('Email subject and body must be strings')
    if len(subject) + len(body) > 5_000_000:
        raise ValueError('Email exceeds the 5,000,000-character CLI input limit')
    if not clean(body):
        raise ValueError('Email body must contain nonempty visible text')
    combined = subject + '\n' + body
    return pd.DataFrame(
        [dict(text=clean(combined), subject=clean(subject), url_count=len(urls(combined)))]
    )


prepare = email_frame


# Loading stage: load this task’s saved model and check its feature schema.
def load(root):
    config, schema, path = metadata(root, 'email')
    model = read_artifact(path)
    validate_model(model, (LinearSVC, LogisticRegression), config)
    transformer = read_artifact(root / 'models/feature_engineering/email_transformer.joblib')
    if transformer.schema() != schema:
        raise ValueError('Saved email transformer and feature schema disagree')
    if model.n_features_in_ != transformer.n_features_:
        raise ValueError('Model and transformer feature counts disagree')
    return model, transformer, config


# Prediction stage: use the prepared features and saved artifacts without fitting.
def predict(frame, loaded):
    model, transformer, config = loaded
    return result(transformer.transform(frame), model, config, RISK_WIDTH)


# Command stage: read options and run this file’s operation.
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Email inference: prepare subject and body, load saved artifacts and predict with the fixed threshold.'
    )
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--subject', default='')
    parser.add_argument('--body', required=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        frame = prepare(args.subject, args.body)
        display(predict(frame, load(args.root.resolve())), args.json)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
