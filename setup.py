"""
Setup script for EmoIA - Emotional Intelligence AI
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="emoia",
    version="3.0.0",
    author="EmoIA Team",
    author_email="contact@emoia.ai",
    description="Intelligence Artificielle Émotionnelle - AI with emotional understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emoia/emoia",
    packages=find_packages(exclude=["tests", "tests.*", "frontend", "frontend.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "production": [
            "psycopg2-binary>=2.9.9",
            "gunicorn>=21.2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "emoia-api=src.core.api:main",
            "emoia-cli=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)