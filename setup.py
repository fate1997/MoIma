#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    author="Gaopeng Ren",
    description="Package for molecule generation.",
    name='moima',
    packages=find_packages(include=['moima', 'moima.*', 'moima.*.*']),
    include_package_data=True,
    url='https://github.com/fate1997/moima',
    version='0.0.1',
)