# setup.py
# Setup installation for the application

from pathlib import Path
from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

test_packages = [
    "great-expectations",
    "pytest",
    "pytest-cov",
]

dev_packages = ["black", "flake8", "isort", "jupyterlab", "pre-commit", "mypy"]

docs_packages = [
    "mkdocs",
    "mkdocs-macros-plugin",
    "mkdocs-material",
    "mkdocstrings",
]

setup(
    name="dogsvscats",
    version="0.6.0",
    license="Apache",
    description="Solution for Kaggle competition https://www.kaggle.com/c/dogs-vs-cats",
    author="Alberto Burgos",
    author_email="albertoburgosplaza@gmail.com",
    keywords=[
        "machine-learning",
        "artificial-intelligence",
        "computer-vision" "kaggle",
        "pytorch",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8.5",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages + docs_packages,
        "docs": docs_packages,
    },
)
