"""Manual offline email/URL inference; no fitting, network access, or dataset reads.

Provisional risk bands (not calibrated risk estimates):
  Email: LOW below threshold-0.5; HIGH at/above threshold+0.5; MEDIUM otherwise.
  URL: LOW below threshold-0.10; HIGH at/above threshold+0.10; MEDIUM otherwise.
Classification always uses the saved threshold, independently of these bands.
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from src.features.url_features import URLFeatures
from src.preprocessing.emails import clean, urls
from src.training.pipeline import scores

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
RISK_BANDS = {'email': 0.5, 'url': 0.10}


def risk_level(kind: str, score: float, threshold: float) -> str:
    """Native-score proximity heuristic, never a calibrated probability mapping."""
    width = RISK_BANDS[kind]
    if score < threshold - width:
        return 'LOW'
    if score >= threshold + width:
        return 'HIGH'
    return 'MEDIUM'


def email_frame(subject: str, body: str) -> pd.DataFrame:
    """Build one inference row from subject/body using the training representation.

    Count links before HTML cleaning so href targets remain represented. Reject
    empty visible bodies; return text, subject and url_count, without fitting.
    """
    if not isinstance(subject, str) or not isinstance(body, str):
        raise ValueError('Email subject and body must be strings')
    if len(subject) + len(body) > 5_000_000:
        raise ValueError('Email exceeds the 5,000,000-character CLI input limit')
    if not clean(body):
        raise ValueError('Email body must contain nonempty visible text')
    combined = subject + '\n' + body
    return pd.DataFrame([dict(text=clean(combined), subject=clean(subject),
                              url_count=len(urls(combined)))])


class Predictor:
    """Lazy-load each trusted local model/transformer once for repeated inference."""
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self._loaded = {}

    @staticmethod
    def _read_artifact(path):
        try:
            return joblib.load(path)
        except (OSError, EOFError, ValueError, ImportError, AttributeError, TypeError) as exc:
            raise ValueError(f'Cannot load saved artifact {path}: {exc}') from exc

    def _load(self, kind):
        if kind in self._loaded:
            return self._loaded[kind]
        folder = self.root/'models'/kind
        config = json.loads((folder/'selected_threshold.json').read_text())
        schema = json.loads((self.root/'models/feature_engineering'/f'{kind}s_schema.json').read_text())
        path = folder/'selected_model.joblib'
        if path.resolve() != (self.root/config['artifact']).resolve():
            raise ValueError('Selected model and saved threshold artifact disagree')
        if config['positive_label'] != 1 or config['prediction_rule'] != 'score >= threshold':
            raise ValueError('Unsupported label mapping or decision rule')
        threshold = config['threshold']
        if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
            raise ValueError('Saved threshold must be a finite number')
        model = self._read_artifact(path)
        expected = LinearSVC if kind == 'email' else RandomForestClassifier
        score_type = 'uncalibrated_margin' if kind == 'email' else 'probability'
        if not isinstance(model, expected) or list(model.classes_) != [0, 1] or config['score_type'] != score_type:
            raise ValueError('Model type, classes or score type are incompatible')
        if kind == 'email':
            transformer = self._read_artifact(self.root/'models/feature_engineering/email_transformer.joblib')
            if transformer.schema() != schema:
                raise ValueError('Saved email transformer and feature schema disagree')
            n_features = transformer.n_features_
        else:
            transformer = URLFeatures()
            if transformer.schema() != schema or not 0 <= threshold <= 1:
                raise ValueError('URL schema or probability threshold is incompatible')
            n_features = len(schema['columns'])
            if list(model.feature_names_in_) != schema['columns']:
                raise ValueError('URL model feature order differs from saved schema')
        if model.n_features_in_ != n_features:
            raise ValueError('Model and transformer feature counts disagree')
        self._loaded[kind] = model, transformer, config
        return self._loaded[kind]

    def _result(self, kind, features, model, config):
        with threadpool_limits(limits=2):
            values, score_type = scores(model, features)
        score = float(values[0])
        threshold = float(config['threshold'])
        return dict(prediction='PHISHING' if score >= threshold else 'LEGITIMATE',
                    score=score, threshold=threshold, risk_level=risk_level(kind, score, threshold),
                    model=config['model'], score_type=score_type,
                    score_note=('Uncalibrated SVM margin, NOT a probability.' if kind == 'email' else
                                'Model-estimated phishing probability, NOT guaranteed real-world risk.'),
                    risk_note=f'Provisional native-score bands: LOW < threshold - {RISK_BANDS[kind]}; HIGH >= threshold + {RISK_BANDS[kind]}; MEDIUM otherwise. Not calibrated.')

    def predict_email(self, subject: str, body: str) -> dict:
        """Transform one message with frozen TF-IDF/scaling and apply its saved margin threshold."""
        frame = email_frame(subject, body)
        model, transformer, config = self._load('email')
        return self._result('email', transformer.transform(frame), model, config)

    def predict_url(self, url: str) -> dict:
        """Extract the thirteen ordered lexical features offline and apply the saved threshold."""
        if not isinstance(url, str) or not url.strip():
            raise ValueError('URL must be a nonempty string with an http:// or https:// scheme')
        if len(url) > 65536:
            raise ValueError('URL exceeds the 65,536-character CLI input limit')
        # Validation and lexical feature extraction use the original URL string.
        features = URLFeatures().transform([url])
        model, _, config = self._load('url')
        return self._result('url', features, model, config)


def display(result, as_json=False):
    if as_json:
        print(json.dumps(result, allow_nan=False))
    else:
        print(f"Prediction: {result['prediction']}\nModel score: {result['score']:.6f} ({result['score_type']})\n"
              f"Decision threshold: {result['threshold']:.6f}\nRisk level: {result['risk_level']}")
        print(result['score_note'])
        print(result['risk_note'])


def interactive(predictor, as_json=False):
    print('Offline inference. Risk bands are provisional; no URLs are visited.')
    while True:
        choice = input('\nChoose email / url / quit: ').strip().lower()
        if choice in ('quit', 'q', 'exit'):
            return
        try:
            if choice in ('email', 'e'):
                subject = input('Subject: ')
                print('Body (finish with a single . on its own line):')
                lines = []; size = len(subject)
                while True:
                    line = input()
                    if line == '.': break
                    size += len(line) + 1
                    if size > 5_000_000:
                        raise ValueError('Email exceeds the input limit; try again')
                    lines.append(line)
                result = predictor.predict_email(subject, '\n'.join(lines))
            elif choice in ('url', 'u'):
                result = predictor.predict_url(input('URL: '))
            else:
                print('Choose email, url or quit.'); continue
            display(result, as_json)
        except (ValueError, OSError, KeyError, AssertionError) as exc:
            print(f'Error: {exc}', file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='Repository containing trusted local artifacts')
    parser.add_argument('--interactive', action='store_true', help='Prompt repeatedly; also the default without a subcommand')
    parser.add_argument('--json', action='store_true', help='Machine-readable prediction output')
    commands = parser.add_subparsers(dest='kind')
    email = commands.add_parser('email', help='Predict from subject and body')
    email.add_argument('--subject', default='')
    email.add_argument('--body', required=True)
    url = commands.add_parser('url', help='Predict a URL without visiting it')
    url.add_argument('--url', required=True)
    # Accept --json after a subcommand too, without overwriting the global flag.
    for command in (email, url):
        command.add_argument('--json', action='store_true', default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.interactive and args.kind:
        parser.error('--interactive cannot be combined with email/url subcommands')
    predictor = Predictor(args.root)
    try:
        if args.kind == 'email':
            display(predictor.predict_email(args.subject, args.body), args.json)
        elif args.kind == 'url':
            display(predictor.predict_url(args.url), args.json)
        else:
            interactive(predictor, args.json)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    except EOFError:
        return
    except KeyboardInterrupt:
        parser.exit(130, '\nInterrupted.\n')

if __name__ == '__main__':
    main()
