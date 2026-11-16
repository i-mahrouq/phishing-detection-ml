"""Lightweight offline compatibility check; does not access datasets or fit models."""
from pathlib import Path
import json

from src.inference.predict import Predictor


def main() -> None:
    predictor = Predictor(Path(__file__).resolve().parents[1])
    results = {
        'email': predictor.predict_email('Meeting', 'Project meeting tomorrow at ten.'),
        'url': predictor.predict_url('https://example.com/'),
    }
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
