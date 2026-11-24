# Tests: check web behavior using saved artifacts or temporary inputs.

import http.client
import json
import threading
from http.server import HTTPServer

import pytest

from src.inference.common import DEFAULT_ROOT
from src.inference.predict import Predictor
from src.web.app import make_handler


# Test setup: run a temporary local server and close it after the test.
@pytest.fixture
def http_request():
    server = HTTPServer(('127.0.0.1', 0), make_handler(DEFAULT_ROOT))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Request stage: send a test HTTP request and read its response.
    def send(method, path, data=None, headers=None):
        connection = http.client.HTTPConnection(*server.server_address, timeout=30)
        try:
            connection.request(method, path, body=data, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    yield send
    server.shutdown()
    thread.join()
    server.server_close()


# Test stage: page and input errors.
def test_page_and_input_errors(http_request):
    status, page = http_request('GET', '/')
    assert status == 200 and b'data-kind="email"' in page and b'data-kind="url"' in page
    assert http_request('GET', '/models/email/selected_model.joblib')[0] == 404
    headers = {'Content-Type': 'application/json'}
    for body in ('{', '[]', '{}', '{"body":123}'):
        status, result = http_request('POST', '/api/email', body, headers)
        assert status == 400 and 'error' in json.loads(result)
    assert (
        http_request('POST', '/api/url', '{}', {**headers, 'Origin': 'https://other.example'})[0]
        == 403
    )


# Test stage: http predictions match existing inference.
@pytest.mark.parametrize(
    'kind,data',
    [
        ('email', {'subject': 'Meeting', 'body': 'Project meeting tomorrow at ten.'}),
        ('url', {'url': 'https://www.example.com'}),
    ],
)
def test_http_predictions_match_existing_inference(http_request, kind, data):
    status, body = http_request(
        'POST', '/api/' + kind, json.dumps(data), {'Content-Type': 'application/json'}
    )
    assert status == 200
    expected = getattr(Predictor(), 'predict_' + kind)(**data)
    actual = json.loads(body)
    assert actual.pop('score') == pytest.approx(expected.pop('score'), abs=1e-12)
    assert actual == expected
