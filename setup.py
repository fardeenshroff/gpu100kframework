# setup.py - Enterprise Python Package Setup
"""
GPU Cluster Optimizer - Research Integration
Professional implementation of NeoCPU and TVM research for large-scale GPU clusters
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gpu-cluster-optimizer",
    version="0.1.0",
    author="Fardeen Fayyaz Shroff",
    author_email="fshroff95@gmail.com",
    description="Research-backed GPU cluster optimization for 100K+ GPU deployments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fardeenshroff/gpu100kframework",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "performance": [
            "psutil>=5.9.0",
            "memory-profiler>=0.60.0",
            "line-profiler>=4.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpu-cluster-optimizer=gpu_cluster_optimizer.cli:main",
            "gco-benchmark=gpu_cluster_optimizer.benchmarking.cli:main",
            "gco-monitor=gpu_cluster_optimizer.monitoring.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# ===============================
# requirements.txt
# ===============================
"""
# Core Dependencies
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
pydantic>=1.10.0
typer>=0.7.0
rich>=12.0.0
loguru>=0.6.0

# Configuration Management
pyyaml>=6.0
python-dotenv>=0.19.0
jsonschema>=4.0.0

# Async and Networking
asyncio>=3.4.3
aiohttp>=3.8.0
websockets>=10.0

# Database and Storage
sqlalchemy>=1.4.0
redis>=4.0.0

# Monitoring and Metrics
prometheus-client>=0.15.0
opencensus>=0.11.0

# Scientific Computing
scikit-learn>=1.1.0
networkx>=2.8.0

# Utilities
click>=8.0.0
tqdm>=4.64.0
colorama>=0.4.5
"""

# ===============================
# requirements-dev.txt  
# ===============================
"""
# Development Dependencies
-r requirements.txt

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-benchmark>=4.0.0
pytest-asyncio>=0.20.0
pytest-mock>=3.10.0
factory-boy>=3.2.0

# Code Quality
black>=22.0.0
isort>=5.10.0
flake8>=5.0.0
mypy>=0.991
bandit>=1.7.0
safety>=2.0.0

# Pre-commit Hooks
pre-commit>=2.20.0

# Documentation
sphinx>=5.0.0
sphinx-rtd-theme>=1.0.0
sphinx-autodoc-typehints>=1.19.0
myst-parser>=0.18.0

# Performance Profiling
psutil>=5.9.0
memory-profiler>=0.60.0
line-profiler>=4.0.0
py-spy>=0.3.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Development Tools
ipython>=8.0.0
jupyter>=1.0.0
ipdb>=0.13.0
"""

# ===============================
# pyproject.toml
# ===============================
"""
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gpu-cluster-optimizer"
version = "0.1.0"
description = "Research-backed GPU cluster optimization for 100K+ GPU deployments"
readme = "README.md"
authors = [{name = "Fardeen Fayyaz Shroff", email = "fshroff95@gmail.com"}]
license = {text = "MIT"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
requires-python = ">=3.8"

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["gpu_cluster_optimizer"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests", 
    "performance: marks tests as performance tests",
    "scalability: marks tests as scalability tests",
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "setup.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
"""
