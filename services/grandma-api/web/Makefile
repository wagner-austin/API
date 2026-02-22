.PHONY: check lint typecheck test build serve dev

check: lint typecheck test

lint:
	npm run lint

typecheck:
	npm run typecheck

test:
	npm run test:unit

build:
	npm run build

serve: build
	python -c "import http.server,ssl;s=http.server.HTTPServer(('0.0.0.0',8091),http.server.SimpleHTTPRequestHandler);c=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);c.load_cert_chain('cert.pem','key.pem');s.socket=c.wrap_socket(s.socket,server_side=True);print('HTTPS on https://0.0.0.0:8091');s.serve_forever()"

dev: build serve
