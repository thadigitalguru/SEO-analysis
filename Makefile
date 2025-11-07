PYTHON := python
PIP := $(PYTHON) -m pip

.PHONY: venv install dev lint test format

venv:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && $(PIP) install --upgrade pip

install:
	. .venv/bin/activate && $(PIP) install -r requirements.txt

# Optional: use pyproject for dev deps
.dev: 
	. .venv/bin/activate && $(PIP) install -e .[dev]

lint:
	. .venv/bin/activate && ruff check .

format:
	. .venv/bin/activate && ruff check . --fix

test:
	. .venv/bin/activate && pytest -q
