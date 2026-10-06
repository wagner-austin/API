include ../../../scripts/make/shell.mk

.PHONY: check _check-unbudgeted install lint typecheck test build

check:
	$(PYTHON) ../../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint typecheck test

install:
	npm ci

lint:
	npm run lint

typecheck:
	npm run typecheck

test:
	npm run test:unit

build:
	npm run build
