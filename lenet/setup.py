# File setup.py
from setuptools import find_packages, setup

setup(
  name="lenet5",
  packages=find_packages(where="src"),
  package_dir={"": "src"},
)
