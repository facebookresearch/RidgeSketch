"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import setup, find_packages

# Read README and requirements
with open("README.md", encoding="utf8") as f:
    readme = f.read()
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RidgeSketch",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    long_description=readme,
    install_requires=requirements,
)
