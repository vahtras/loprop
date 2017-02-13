#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="LoProp",
    version="0.2.1",
    packages=["loprop"],
    #packages=find_packages(exclude=("tests",)),
    scripts=['scripts/loprop'],
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    install_requires = ["daltools"],
    description = 'LoProp implementation for Dalton',
    )
