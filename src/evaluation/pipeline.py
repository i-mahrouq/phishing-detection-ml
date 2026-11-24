# Evaluation commands: forward existing commands to the separate email and URL evaluators.

import argparse
import json
from pathlib import Path
from importlib import import_module
from threadpoolctl import threadpool_limits
from .common import binary_metrics, finite, save_errors, plot_matrix, load_selected


# External evaluation stage: measure detected and missed phishing URLs only.
def external_metrics(y, values, threshold):
    return task('url').external_metrics(y, values, threshold)


# Dispatch stage: choose the requested task module and reject unknown tasks.
def task(kind):
    if kind not in ('email', 'url'):
        raise ValueError(f'Unknown evaluation task: {kind}')
    return import_module('.evaluate_' + kind, __package__)


# Provenance stage: record the model, threshold and test inputs that must remain unchanged.
def frozen_paths(root, kind=None):
    result = {}
    for name in ((kind,) if kind else ('email', 'url')):
        result.update(task(name).frozen_paths(root))
    return result


# Loading stage: read permitted held-out features and verify their row alignment.
def load_data(root, kind, part='test'):
    return task(kind).load_data(root, part)


# Leakage check: compare this task’s training and test groups.
def overlap(root, kind, rows):
    return task(kind).overlap(root, rows)


# Verification stage: check saved evaluation evidence without changing it.
def verify(root, kind=None):
    for name in ((kind,) if kind else ('email', 'url')):
        task(name).verify(root)
    return dict(completed=True, models_and_thresholds_unchanged=True, reports_verified=True)


# Evaluation stage: score frozen held-out data or verify completed results.
def run(root, kind=None):
    for name in ((kind,) if kind else ('email', 'url')):
        task(name).run(root)


# Command stage: read options and run this file’s operation.
def main():
    parser = argparse.ArgumentParser(
        description='Evaluation commands: forward existing commands to the separate email and URL evaluators.'
    )
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--kind', choices=('email', 'url'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.verify:
            print(json.dumps(verify(args.root, args.kind)))
        else:
            run(args.root, args.kind)


if __name__ == '__main__':
    main()
