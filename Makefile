PROJECT_NAME = policy-diff-assist
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
APP = src/policy_diff_assist/main.py

.PHONY: help init install run test lint format clean reset

# Default help target
help:
	@echo "Available commands:"
	@echo "  make init       - Create venv and install dependencies"
	@echo "  make install    - Install dependencies (venv must exist)"
	@echo "  make run        - Run the application"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"
	@echo "  make clean      - Remove venv and cache files"
	@echo "  make reset      - Complete reset of venv"

init: $(VENV)/bin/activate
	@echo "✓ Virtual environment ready at $(VENV)"

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV) --prompt $(PROJECT_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Virtual environment created and dependencies installed!"

install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

run: $(VENV)/bin/activate
	$(PYTHON) $(APP)

test: $(VENV)/bin/activate
	$(PYTHON) -m pytest

lint: $(VENV)/bin/activate
	$(PYTHON) -m ruff check .

format: $(VENV)/bin/activate
	$(PYTHON) -m ruff format .

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache *.egg-info
	rm -rf .coverage htmlcov .mypy_cache

reset: clean init install
	@echo "✓ Reset complete!"