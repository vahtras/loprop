#!/usr/bin/env python

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
import os

#if submodules have not been initiated, just update them
os.system( 'git submodule update --init --recursive' )


setup(name="LoProp",
    version="0.1.6",
    packages = ["loprop"],
    scripts = ['scripts/loprop'],
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    install_requires = ["daltools"],
    description = 'LoProp implementation for Dalton',
    )
