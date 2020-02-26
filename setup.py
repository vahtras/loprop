#!/usr/bin/env python

from setuptools import setup

setup(
    name="LoProp",
    version="0.2.4",
    packages=["loprop"],
    entry_points={
        'console_scripts': ['loprop=loprop.__main__:main'],
    },
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    install_requires=["h5py", "daltools"],
    description='LoProp implementation for Dalton',
    )
