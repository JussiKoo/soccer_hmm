from setuptools import setup, find_packages

setup(
    author="Jussi Kauppinen",
    description="Bayesian HMM modeling for multivariate time series data."
    name="soccer_hmm",
    version="0.1.0",
    packages=find_packages(include=["soccer_hmm", "soccer_hmm.*"])
)