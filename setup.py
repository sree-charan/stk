from setuptools import setup, find_packages

setup(
    name="stock-chat",
    version="1.2.3",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0", "pandas>=2.0.0", "scipy>=1.10.0",
        "xgboost>=2.0.0", "torch>=2.0.0", "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0", "optuna>=3.0.0",
        "fastapi>=0.100.0", "uvicorn>=0.23.0", "websockets>=11.0", "pydantic>=2.0.0",
        "yfinance>=0.2.0", "fredapi>=0.5.0", "newsapi-python>=0.2.7",
        "requests>=2.31.0", "vaderSentiment>=3.3.2",
        "click>=8.1.0", "rich>=13.0.0",
    ],
    entry_points={"console_scripts": ["stk=cli.main:main"]},
    package_data={"cli": ["py.typed"]},
    python_requires=">=3.10",
)
