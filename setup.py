"""
setup.py — Building Height Analysis: Dhaka City
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dhaka-height-analysis",
    version="1.0.0",
    author="Urban AI Lab",
    author_email="urbanai@example.com",
    description="Building height estimation using deep learning for Dhaka City",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/dhaka-height-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "dhaka-pipeline=pipeline.run_pipeline:main",
            "dhaka-dashboard=dashboard.app:main",
        ],
    },
)
