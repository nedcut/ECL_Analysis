# Makefile for Brightness Sorcerer testing and development

.PHONY: help test test-unit test-integration test-gui test-fast test-coverage test-report clean install-test-deps setup-test-data lint format

help:		## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-test-deps:	## Install test dependencies
	pip install -r requirements-test.txt

setup-test-data:	## Create test fixtures and sample data
	python tests/fixtures/create_test_data.py

test:		## Run all tests
	pytest

test-unit:	## Run only unit tests
	pytest tests/unit/ -v

test-integration:	## Run only integration tests
	pytest tests/integration/ -v

test-gui:	## Run GUI tests (requires display)
	pytest -m gui -v

test-fast:	## Run fast tests (skip slow ones)
	pytest -m "not slow" -v

test-coverage:	## Run tests with coverage report
	pytest --cov=brightness_sorcerer --cov=main --cov-report=html --cov-report=term-missing

test-report:	## Generate detailed test report
	pytest --html=test-report.html --self-contained-html --junitxml=test-results.xml

test-smoke:	## Run smoke tests for quick validation
	pytest -m smoke -v

test-performance:	## Run performance benchmarks
	pytest -m performance --benchmark-only

clean:		## Clean up test artifacts
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf test-report.html
	rm -rf test-results.xml
	rm -rf coverage.xml
	rm -rf .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:		## Run code linting
	@echo "Running code analysis..."
	@python -m py_compile main.py
	@python -c "import brightness_sorcerer; print('✅ Package imports successfully')"

format:		## Format code (placeholder for future formatter)
	@echo "Code formatting not configured yet"

# Testing shortcuts
unit: test-unit
integration: test-integration
coverage: test-coverage
fast: test-fast
smoke: test-smoke

# Development workflow
dev-setup: install-test-deps setup-test-data	## Complete development setup
	@echo "✅ Development environment set up"

ci-test:	## Run CI-friendly tests
	pytest --junitxml=test-results.xml --cov=brightness_sorcerer --cov=main --cov-report=xml

# Quality assurance
qa: lint test-coverage		## Run full quality assurance suite
	@echo "✅ Quality assurance complete"