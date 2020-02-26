#!/usr/bin/env python

from setuptools import setup

from loprop import __version__

setup(
    name="LoProp",
    version=__version__,
    packages=["loprop"],
    entry_points={
        'console_scripts': ['loprop=loprop.__main__:main'],
    },
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    install_requires=["h5py", "daltools"],
    description='LoProp implementation for Dalton',
    )
