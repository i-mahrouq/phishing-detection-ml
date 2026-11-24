# Shared features: count suspicious keywords, describe input files and save checked reports.

import re

KEYWORDS = (
    'urgent',
    'verify',
    'verification',
    'password',
    'account',
    'login',
    'signin',
    'security',
    'suspend',
    'invoice',
    'payment',
    'bank',
    'confirm',
    'credential',
    'wallet',
    'winner',
    'click',
)
WORDS = re.compile(r'\b(?:' + '|'.join(KEYWORDS) + r')\b', re.I)
FRAGMENTS = re.compile('|'.join(KEYWORDS), re.I)


# Keyword stage: count suspicious terms using whole words or URL fragments.
def suspicious_count(text, whole_words=True):
    pattern = WORDS if whole_words else FRAGMENTS
    return sum(1 for _ in pattern.finditer(text))


SPLITS = ('train', 'validation', 'test')


# Provenance stage: record each input file’s path, size, timestamp and row count.
def manifest(paths):
    import pyarrow.parquet as pq

    inputs = {}
    for name, path in paths.items():
        details = path.stat()
        inputs[name] = {
            'path': str(path),
            'size': details.st_size,
            'mtime_ns': details.st_mtime_ns,
            'rows': pq.read_metadata(path).num_rows,
        }
    return inputs


# Reporting stage: save the feature report only if its output checks pass.
def publish_report(path, report, verify):
    import json

    path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        report['validation'] = verify()
    except Exception:
        path.unlink(missing_ok=True)
        raise
    path.write_text(json.dumps(report, indent=2) + '\n')


# Command stage: run feature generation or verify its existing outputs.
def pipeline_main(run, verify, description):
    import argparse
    import json
    import logging
    from pathlib import Path

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.verify:
        print(json.dumps(verify(args.root)))
    else:
        run(args.root, args.seed)
