import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
README = os.path.join(HERE, "README.md")
LONG_DESC = open(README, encoding="utf-8").read() if os.path.exists(README) else ""

setup(
    name="ai-plus",
    version="0.4.0",
    description="AI+ — Assistente AI ibrido con routing intelligente locale (Ollama) e cloud (OpenAI, OpenRouter, Gemini)",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="AI+ Team",
    author_email="team@ai-plus.dev",
    url="https://github.com/ai-plus/ai-plus",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    python_requires=">=3.10",

    install_requires=[
        "click>=8.0",
        "rich>=13.0",
        "requests>=2.28",
        "urllib3>=2.0",
        "pyyaml>=6.0",
        "flask>=2.3",
    ],

    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
        "server": [
            "flask>=2.3",
        ],
        "knowledge": [
            "numpy>=1.24",
        ],
        "voice": [
            "pyaudio>=0.2",
        ],
        "all": [
            "flask>=2.3",
            "numpy>=1.24",
        ],
    },

    entry_points={
        "console_scripts": [
            "ai-plus=hycoder.cli:main",
            "hy=hycoder.cli:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
)
