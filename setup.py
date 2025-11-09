# setup.py - SIMPLE VERSION
from setuptools import setup

setup(
    name="resumate",
    version="1.0",
    packages=["src.parsing"],
    package_dir={"src.parsing": "src/parsing"},
)