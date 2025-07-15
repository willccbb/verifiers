.PHONY: test test-v test-cov test-parser test-env test-rubric clean help

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

test:  ## Run all tests (quiet mode)
	pytest

test-v:  ## Run all tests with verbose output
	pytest -v

test-cov:  ## Run tests with coverage report
	pytest --cov=verifiers --cov-report=term-missing --cov-report=html

test-parser:  ## Run parser tests only
	pytest tests/test_parser.py tests/test_xml_parser.py tests/test_think_parser.py -v

test-env:  ## Run environment tests only
	pytest tests/test_environment.py tests/test_singleturn_env.py tests/test_multiturn_env.py tests/test_env_group.py -v

test-rubric:  ## Run rubric tests only
	pytest tests/test_rubric.py tests/test_rubric_group.py -v

test-failed:  ## Re-run only failed tests from last run
	pytest --lf -v

test-debug:  ## Run tests with debugging output (no capture)
	pytest -vvs

test-parallel:  ## Run tests in parallel (requires pytest-xdist)
	@command -v pytest-xdist >/dev/null 2>&1 || pip install pytest-xdist
	pytest -n auto

clean:  ## Clean test artifacts
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete