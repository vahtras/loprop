#!/usr/bin/env python

from setuptools import setup

setup(name="LoProp",
    version="0.1.6",
    packages = ["loprop"],
    scripts = ['scripts/loprop'],
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    install_requires = ["daltools", "util"],
    description = 'LoProp implementation for Dalton',
    )
