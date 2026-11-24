# Local web server: serve the page and connect its email and URL forms to saved-model inference.

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from src.inference.common import DEFAULT_ROOT
from src.inference import email, url


# Server setup: connect the local page and prediction endpoints to cached task models.
def make_handler(root):
    loaded = {}
    modules = {'email': email, 'url': url}
    page = Path(__file__).with_name('index.html').read_bytes()

    # Request handler: serve the page and the two prediction endpoints.
    class Handler(BaseHTTPRequestHandler):
        # Response stage: send an HTML page or JSON result with its HTTP status.
        def respond(self, status, content, content_type='application/json'):
            if content_type == 'application/json':
                content = json.dumps(content, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        # Page stage: serve the local form page and reject other paths.
        def do_GET(self):
            if self.path == '/':
                self.respond(200, page, 'text/html; charset=utf-8')
            else:
                self.respond(404, {'error': 'Page not found'})

        # Prediction stage: validate the request and send it to email or URL inference.
        def do_POST(self):
            kind = {'/api/email': 'email', '/api/url': 'url'}.get(self.path)
            if kind is None:
                self.respond(404, {'error': 'Endpoint not found'})
                return

            origin = self.headers.get('Origin')
            expected_origin = f'http://{self.headers.get("Host")}'
            if origin and origin != expected_origin:
                self.respond(403, {'error': 'Use the local project page to submit input'})
                return
            try:
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('Send application/json')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 20_100_000:
                    raise ValueError('Request is empty or too large')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Expected a JSON object')
                module = modules[kind]
                features = (
                    module.prepare(data.get('subject', ''), data.get('body', ''))
                    if kind == 'email'
                    else module.prepare(data.get('url', ''))
                )
            except (ValueError, UnicodeError) as exc:
                self.respond(400, {'error': str(exc)})
                return
            try:
                if kind not in loaded:
                    loaded[kind] = module.load(root)
                self.respond(200, module.predict(features, loaded[kind]))
            except (OSError, ValueError, KeyError, AssertionError) as exc:
                self.respond(503, {'error': f'Inference unavailable: {exc}'})

    return Handler


# Command stage: read options and run this file’s operation.
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Local web server: serve the page and connect its email and URL forms to saved-model inference.'
    )
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    with HTTPServer(('127.0.0.1', args.port), make_handler(args.root.resolve())) as server:
        print(f'Open http://127.0.0.1:{server.server_port} (Ctrl+C to stop)', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
