# setup.py
# Setup installation for the application

from setuptools import find_namespace_packages, setup
import os

BASE_DIR = os.environ.get("BASE_DIR", None)
BASE_DIR = "." if BASE_DIR is None else BASE_DIR

# Load packages from requirements.txt
with open(f"{BASE_DIR}/requirements.txt") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

test_packages = [
    "great-expectations",
    "pytest",
    "pytest-cov",
]

dev_packages = [
    "black",
    "flake8",
    "isort",
    "jupyterlab",
    "pre-commit",
    "mypy",
    "pylint",
]

docs_packages = [
    "mkdocs",
    "mkdocs-macros-plugin",
    "mkdocs-material",
    "mkdocstrings",
]

setup(
    name="dogsvscats",
    version="0.6.6",
    license="Apache",
    description="Solution for Kaggle competition https://www.kaggle.com/c/dogs-vs-cats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alberto Burgos",
    author_email="albertoburgosplaza@gmail.com",
    url="https://github.com/albertoburgosplaza/dogs-vs-cats",
    keywords=[
        "machine-learning",
        "artificial-intelligence",
        "computer-vision",
        "kaggle",
        "pytorch",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8.10",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages + docs_packages,
        "docs": docs_packages,
    },
)
