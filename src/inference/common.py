# Shared inference: validate artifacts, calculate native scores and format prediction results.

import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from src.training.common import scores

DEFAULT_ROOT = Path(__file__).resolve().parents[2]


# Display stage: assign a provisional band using the supplied width around the threshold.
def risk_level(score: float, threshold: float, width: float) -> str:
    if score < threshold - width:
        return 'LOW'
    if score >= threshold + width:
        return 'HIGH'
    return 'MEDIUM'


# Loading stage: read a trusted saved object and explain loading failures.
def read_artifact(path):
    try:
        return joblib.load(path)
    except (OSError, EOFError, ValueError, ImportError, AttributeError, TypeError) as exc:
        raise ValueError(f'Cannot load saved artifact {path}: {exc}') from exc


# Validation stage: check saved paths, label mapping, schema and decision threshold.
def metadata(root, kind):
    folder = root / 'models' / kind
    config = json.loads((folder / 'selected_threshold.json').read_text())
    schema = json.loads((root / 'models/feature_engineering' / f'{kind}s_schema.json').read_text())
    path = folder / 'selected_model.joblib'
    if path.resolve() != (root / config['artifact']).resolve():
        raise ValueError('Selected model and saved threshold artifact disagree')
    if config['positive_label'] != 1 or config['prediction_rule'] != 'score >= threshold':
        raise ValueError('Unsupported label mapping or decision rule')
    threshold = config['threshold']
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError('Saved threshold must be a finite number')
    return config, schema, path


# Validation stage: check the retained model type, classes and score format.
def validate_model(model, expected, config):
    score_type = 'probability' if hasattr(model, 'predict_proba') else 'uncalibrated_margin'
    if (
        not isinstance(model, expected)
        or list(model.classes_) != [0, 1]
        or config['score_type'] != score_type
    ):
        raise ValueError('Model type, classes or score type are incompatible')
    if score_type == 'probability' and not 0 <= config['threshold'] <= 1:
        raise ValueError('Probability threshold must be between 0 and 1')


# Prediction stage: apply the saved threshold and return the score, label and display notes.
def result(features, model, config, band_width):
    with threadpool_limits(limits=2):
        values, score_type = scores(model, features)
    score = float(values[0])
    threshold = float(config['threshold'])
    return dict(
        prediction='PHISHING' if score >= threshold else 'LEGITIMATE',
        score=score,
        threshold=threshold,
        risk_level=risk_level(score, threshold, band_width),
        model=config['model'],
        score_type=score_type,
        score_note=(
            'Uncalibrated SVM margin, NOT a probability.'
            if score_type == 'uncalibrated_margin'
            else 'Model-estimated phishing probability, NOT guaranteed real-world risk.'
        ),
        risk_note=f'Provisional native-score bands: LOW < threshold - {band_width}; HIGH >= threshold + {band_width}; MEDIUM otherwise. Not calibrated.',
    )


# Output stage: print the prediction as JSON or readable text.
def display(result, as_json=False):
    if as_json:
        print(json.dumps(result, allow_nan=False))
    else:
        print(
            f"Prediction: {result['prediction']}\nModel score: {result['score']:.6f} ({result['score_type']})\n"
            f"Decision threshold: {result['threshold']:.6f}\nRisk level: {result['risk_level']}"
        )
        print(result['score_note'])
        print(result['risk_note'])
