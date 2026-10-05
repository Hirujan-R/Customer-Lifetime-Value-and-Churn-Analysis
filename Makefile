.DEFAULT_GOAL := help
UV ?= uv
PIPELINE ?= data_ingestion

.PHONY: help install sync ingest run test lint format check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: sync ## Alias for sync

sync: ## Create/refresh the uv environment from uv.lock
	$(UV) sync

ingest: ## Run the data ingestion pipeline
	$(UV) run kedro run --pipeline data_ingestion

run: ## Run the full default pipeline
	$(UV) run kedro run

test: ## Run the test suite
	$(UV) run pytest

lint: ## Lint with ruff
	$(UV) run ruff check src tests

format: ## Format with ruff
	$(UV) run ruff format src tests

check: lint test ## Lint and test

clean: ## Remove caches and transient artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
