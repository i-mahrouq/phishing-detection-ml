# Training commands: forward existing commands to the separate email and URL trainers.

import argparse
import json
import logging
from pathlib import Path
from threadpoolctl import threadpool_limits

from importlib import import_module
from .common import (
    PARTS,
    metrics,
    scores,
    selection_key,
    tune_threshold,
    validate_labels,
    write_json,
)

KINDS = ('email', 'url')


# Dispatch stage: choose the requested task module and reject unknown tasks.
def task(kind):
    if kind not in KINDS:
        raise ValueError(f'Unknown training task: {kind}')
    return import_module('.train_' + kind, __package__)


# Configuration stage: create the retained models with their original parameters.
def model_configs(kind, seed=42):
    return task(kind).model_configs(seed)


# Training stage: compare candidates and save the validation-selected model and threshold.
def train_kind(root, kind, seed=42):
    return task(kind).train(root, seed)


# Verification stage: check this task’s saved artifacts and reports.
def verify_kind(root, kind):
    return task(kind).verify_kind(root)


# Comparison stage: forward saved-candidate comparison to the requested task.
def compare_kind(root, kind):
    return task(kind).compare(root)


# Provenance stage: record input sizes and timestamps to detect changes.
def fingerprint(root, kind):
    return task(kind).fingerprint(root)


# Input stage: list the files needed for this task’s training and validation.
def input_files(root, kind):
    return task(kind).input_files(root)


# Validation stage: check feature schemas, row alignment and input file identities.
def preflight(root, kind):
    return task(kind).preflight(root)


# Loading stage: read training or validation features and labels only.
def load_partition(root, kind, part):
    return task(kind).load_partition(root, part)


# Command stage: read options and run this file’s operation.
def main():
    parser = argparse.ArgumentParser(
        description='Training commands: forward existing commands to the separate email and URL trainers.'
    )
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    with threadpool_limits(limits=2):
        if args.verify:
            summary = json.loads((args.root / 'reports/training/training_summary.json').read_text())
            assert summary['completed']
            print(json.dumps([verify_kind(args.root, k) for k in KINDS]))
        else:
            for kind in KINDS:
                train_kind(args.root, kind, args.seed)
            checks = [verify_kind(args.root, k) for k in KINDS]
            write_json(
                args.root / 'reports/training/training_summary.json',
                dict(
                    completed=True,
                    seed=args.seed,
                    validation=checks,
                    test_evaluated=False,
                    external_evaluated=False,
                ),
            )


if __name__ == '__main__':
    main()
