#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    author="Gaopeng Ren",
    description="Package for molecule generation.",
    name='MolGEN',
    packages=find_packages(include=['MolGEN', 'MolGEN.*', 'MolGEN.*.*']),
    include_package_data=True,
    url='https://github.com/fate1997/MolGEN',
    version='0.0.0',
)