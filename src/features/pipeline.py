# Feature commands: forward existing commands to the separate email and URL pipelines.

import argparse
import json
from pathlib import Path
from importlib import import_module
from .common import SPLITS, manifest


# Dispatch stage: choose the requested task module and reject unknown tasks.
def task(kind):
    if kind not in ('email', 'emails', 'url', 'urls'):
        raise ValueError(f'Unknown feature task: {kind}')
    return import_module('.' + kind.rstrip('s') + '_pipeline', __package__)


# Input stage: list the task’s saved splits and external inputs when needed.
def inputs(root, kind):
    return task(kind).inputs(root)


# Verification stage: check this task’s saved artifacts and reports.
def verify_kind(root, kind):
    return task(kind).verify(root)


# Execution stage: forward generation to the requested task module.
def run_kind(root, kind, seed=42):
    return task(kind).run(root, seed)


# Command stage: read options and run this file’s operation.
def main():
    parser = argparse.ArgumentParser(
        description='Feature commands: forward existing commands to the separate email and URL pipelines.'
    )
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kind', choices=('email', 'url'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    for kind in ((args.kind,) if args.kind else ('email', 'url')):
        if args.verify:
            print(json.dumps(verify_kind(args.root, kind)))
        else:
            run_kind(args.root, kind, args.seed)


if __name__ == '__main__':
    main()
