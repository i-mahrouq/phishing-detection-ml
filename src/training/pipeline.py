"""Backward-compatible V1 entry point; prefer train_email or train_url.

The task modules own algorithm constructors; common.py owns shared utilities.
Existing imports and the two-task command remain available for saved research.
"""
import argparse
import json
import logging
from pathlib import Path
from threadpoolctl import threadpool_limits

from .common import (compare_kind, fingerprint, input_files, load_partition, metrics,
                     preflight, scores, selection_key, tune_threshold, validate_labels,
                     verify_kind, write_json)

KINDS = ('email', 'url')
PARTS = ('train', 'validation')


def model_configs(kind, seed=42):
    if kind == 'email':
        from .train_email import model_configs as configurations
    elif kind == 'url':
        from .train_url import model_configs as configurations
    else:
        raise ValueError(f'Unknown training task: {kind}')
    return configurations(seed)


def train_kind(root, kind, seed=42):
    from .common import train_kind as train
    return train(root, kind, seed, model_configs(kind, seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    with threadpool_limits(limits=2):
        if args.verify:
            summary=json.loads((args.root/'reports/training/training_summary.json').read_text())
            assert summary['completed']
            print(json.dumps([verify_kind(args.root,k) for k in KINDS]))
        else:
            for kind in KINDS:
                train_kind(args.root,kind,args.seed)
            checks=[verify_kind(args.root,k) for k in KINDS]
            write_json(args.root/'reports/training/training_summary.json',
                       dict(completed=True,seed=args.seed,validation=checks,test_evaluated=False,external_evaluated=False))

if __name__ == '__main__':
    main()
