#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import os

#if submodules have not been initiated, just update them
os.system( 'git submodule update --init --recursive' )

reqs = ['argparse>=1.2.1',
        'numpy>=1.9.2',
        'scipy>=0.16.0',
        'wsgiref>=0.1.2' ]

setup(name="LoProp",
    version="0.1",
    packages=[".", "test", "daltools", "daltools/util"],
    scripts=["loprop.py",],
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    install_requires = reqs,
    )
