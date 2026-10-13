"""Loopback-only prototype API. Startup does not start any model or training."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .common import ROOT, read_json
from .pipeline import Pipeline


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/api/config':
            return self.send_json(200, read_json(ROOT/'configs/prototype.json'))
        if self.path != '/':
            return self.send_json(404, {'error': 'not_found'})
        data = (ROOT/'web/index.html').read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != '/api/assess':
            return self.send_json(404, {'error': 'not_found'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32_000_000:
                raise ValueError('请求大小不合法')
            request = json.loads(self.rfile.read(length))
            result = Pipeline().run(request)
            self.send_json(200, result)
        except ValueError as exc:
            self.send_json(400, {'error': str(exc)[:240]})
        except Exception:
            self.send_json(503, {'error': '模型或处理服务不可用，请核对本地端点与配置'})


def serve(port=18110):
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
