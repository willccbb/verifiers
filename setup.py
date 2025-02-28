from setuptools import setup, find_packages

setup(
    name="verifiers",
    version="0.1.0",
    packages=find_packages(include=["verifiers", "configs"]),  # Explicitly include both packages
)
