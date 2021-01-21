#!/usr/bin/env python

from setuptools import setup

setup(
    name="LoProp",
    version="0.3.1",
    packages=["loprop"],
    entry_points={
        'console_scripts': ['loprop=loprop.__main__:main'],
    },
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    install_requires=["blocked-matrix-utils"],
    extras_require={
        "dalton": ["daltools"],
        "vlx": ["h5py"],
    },
    description='LoProp implementation for Dalton',
    )
