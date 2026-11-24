# Prediction menu: send email and URL inputs to their independent inference modules.

import argparse
import json
from pathlib import Path
import sys
from importlib import import_module
from .common import DEFAULT_ROOT, display, read_artifact, result
from .email import email_frame


# Dispatch stage: choose the requested task module and reject unknown tasks.
def task(kind):
    if kind not in ('email', 'url'):
        raise ValueError(f'Unknown inference task: {kind}')
    return import_module('.' + kind, __package__)


# Prediction router: cache artifacts separately for each task.
class Predictor:

    # Setup stage: initialize the cached artifacts.
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self._loaded = {}

    _read_artifact = staticmethod(read_artifact)

    # Loading stage: cache this task’s saved artifacts for later predictions.
    def _load(self, kind):
        if kind not in self._loaded:
            self._loaded[kind] = task(kind).load(self.root)
        return self._loaded[kind]

    # Prediction stage: use the task’s display width with the shared result formatter.
    def _result(self, kind, features, model, config):
        return result(features, model, config, task(kind).RISK_WIDTH)

    # Email prediction stage: prepare the message and call email inference.
    def predict_email(self, subject, body):
        module = task('email')
        frame = module.prepare(subject, body)
        return module.predict(frame, self._load('email'))

    # URL prediction stage: prepare the address and call URL inference.
    def predict_url(self, url):
        module = task('url')
        features = module.prepare(url)
        return module.predict(features, self._load('url'))


# Menu stage: read email or URL inputs until the user quits.
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
                lines = []
                size = len(subject)
                while True:
                    line = input()
                    if line == '.':
                        break
                    size += len(line) + 1
                    if size > 5_000_000:
                        raise ValueError('Email exceeds the input limit; try again')
                    lines.append(line)
                result = predictor.predict_email(subject, '\n'.join(lines))
            elif choice in ('url', 'u'):
                result = predictor.predict_url(input('URL: '))
            else:
                print('Choose email, url or quit.')
                continue
            display(result, as_json)
        except (ValueError, OSError, KeyError, AssertionError) as exc:
            print(f'Error: {exc}', file=sys.stderr)


# Command stage: read options and run this file’s operation.
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Prediction menu: send email and URL inputs to their independent inference modules.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=DEFAULT_ROOT,
        help='Repository containing trusted local artifacts',
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Prompt repeatedly; also the default without a subcommand',
    )
    parser.add_argument('--json', action='store_true', help='Machine-readable prediction output')
    commands = parser.add_subparsers(dest='kind')
    email = commands.add_parser('email', help='Predict from subject and body')
    email.add_argument('--subject', default='')
    email.add_argument('--body', required=True)
    url = commands.add_parser('url', help='Predict a URL without visiting it')
    url.add_argument('--url', required=True)

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
