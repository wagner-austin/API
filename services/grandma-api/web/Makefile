include ../../../scripts/make/shell.mk

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

# The HTTPS static server is the api package's scripts/webserver.py (tested
# there), run from that package's venv with the port and the directory to
# serve; it used to be an inline python -c program here, which no portable
# recipe can carry.
serve: build
	poetry -C .. run python -m scripts.webserver 8091 web

dev: build serve
