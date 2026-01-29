"""
Setup configuration for Mental Health Prediction Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mental-health-prediction",
    version="1.0.0",
    author="Muhammad Farooq",
    author_email="mfarooqshafee333@gmail.com",
    description="A comprehensive machine learning project for mental health prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Muhammad-Farooq-13/mental-health-prediction",
    packages=find_packages(exclude=["tests", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "pylint>=2.17.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "mental-health-train=mlops_pipeline:main",
            "mental-health-serve=flask_app:app",
        ],
    },
)
