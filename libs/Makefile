include ../scripts/make/shell.mk

.PHONY: lint test check

# Each target runs the same target in every package directly under this one
# that carries a Makefile, runs them all even after one fails, and names the
# failures at the end (tools/maketools, the ``fan-out`` command).
lint:
	$(PYTHON) ../tools/maketools/scripts/run.py fan-out lint .

test:
	$(PYTHON) ../tools/maketools/scripts/run.py fan-out test .

check:
	$(PYTHON) ../tools/maketools/scripts/run.py fan-out check .
