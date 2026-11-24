# URL inference: validate the address, load saved artifacts and predict from lexical features.

import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features.url_features import URLFeatures
from .common import DEFAULT_ROOT, metadata, read_artifact, validate_model, result, display

RISK_WIDTH = 0.1


# Preparation stage: validate input and build the representation used by the model.
def prepare(url):
    if not isinstance(url, str) or not url.strip():
        raise ValueError('URL must be a nonempty string with an http:// or https:// scheme')
    if len(url) > 65536:
        raise ValueError('URL exceeds the 65,536-character CLI input limit')
    return URLFeatures().transform([url])


# Loading stage: load this task’s saved model and check its feature schema.
def load(root):
    config, schema, path = metadata(root, 'url')
    model = read_artifact(path)
    if isinstance(model, Pipeline) and (
        list(model.named_steps) != ['scaler', 'classifier']
        or not isinstance(model.named_steps['scaler'], StandardScaler)
        or not isinstance(model.named_steps['classifier'], LogisticRegression)
    ):
        raise ValueError('Only the retained scaled URL Logistic Regression pipeline is supported')
    validate_model(model, (RandomForestClassifier, Pipeline), config)
    transformer = URLFeatures()
    if transformer.schema() != schema or not 0 <= config['threshold'] <= 1:
        raise ValueError('URL schema or probability threshold is incompatible')
    if list(model.feature_names_in_) != schema['columns']:
        raise ValueError('URL model feature order differs from saved schema')
    if model.n_features_in_ != len(schema['columns']):
        raise ValueError('Model and transformer feature counts disagree')
    return model, transformer, config


# Prediction stage: use the prepared features and saved artifacts without fitting.
def predict(features, loaded):
    model, _, config = loaded
    return result(features, model, config, RISK_WIDTH)


# Command stage: read options and run this file’s operation.
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='URL inference: validate the address, load saved artifacts and predict from lexical features.'
    )
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--url', required=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        features = prepare(args.url)
        display(predict(features, load(args.root.resolve())), args.json)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
