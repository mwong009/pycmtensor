.PHONY: clean clean-build clean-pyc clean-test docs cleandocs help install lint lint/isort lint/black build
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-test cleandocs ## remove all build, test, coverage docs and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint/black: ## check style with black
	poetry run black .

lint/isort: ## isort the imports
	poetry run isort .

lint: lint/black lint/isort ## check style

test: ## run tests quickly with the default Python
	pytest --cov=pycmtensor --cov-report=html --cov-report=term tests/

docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

cleandocs:
	$(MAKE) -C docs clean
	rm -fr docs/autoapi/

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

install: clean ## install the package to the active Python's site-packages
	poetry install
