SHELL := /bin/bash

# Makefile
.PHONY: help
help:
	@echo "Commands:"
	@echo "venv-dev : creates development environment."
	@echo "style : runs style formatting."
	@echo "clean : cleans all unecessary files."
	@echo "build : create a package from setup.py."
	@echo "testpypi : upload packages to testpypi."

# Conda
.ONESHELL:
conda:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& sh Miniconda3-latest-Linux-x86_64.sh -b -u \
	&& rm -fr Miniconda3-latest-Linux-x86_64.sh \
	&& ~/miniconda3/bin/conda init bash

# Environment
.ONESHELL:
venv-dev:
	conda create -y -n dogsvscats python=3.7 && \
	conda activate dogsvscats && \
	pip install --upgrade pip setuptools wheel && \
	pip install -e ".[dev]" --no-cache-dir

.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	pip install --upgrade pip setuptools wheel && \
	pip install -e . --no-cache-dir

# Styling
.PHONY: style
style:
	black .
	flake8
	isort .

# Cleaning
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage
	rm -fr dist

# Packaging
.PHONY: build
build: clean
	pip install --upgrade build
	BASE_DIR=`pwd` python -m build

# Uploading to testpypi
.PHONY: testpypi
testpypi:
	pip install --upgrade twine
	python -m twine upload --repository testpypi dist/*
