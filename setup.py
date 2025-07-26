"""
Setup file for gpu100kframework
Target: NVIDIA AI Workload Resiliency Engineering
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gpu100kframework",
    version="0.1.0",
    author="Fardeen Fayyaz Shroff",
    author_email="fshroff95@gmail.com",
    description="gpu100kframework - Driving GPU cluster downtime towards zero",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fardeenshroff/gpu100kframework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="ai, gpu, cluster, resiliency, nvidia, monitoring, 100k, fault-tolerance",
    project_urls={
        "Bug Reports": "https://github.com/fardeenshroff/gpu100kframework/issues",
        "Source": "https://github.com/fardeenshroff/gpu100kframework",
        "Documentation": "https://github.com/fardeenshroff/gpu100kframework/docs",
    },
)
