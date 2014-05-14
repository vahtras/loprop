#!/usr/bin/env python
from distutils.core import setup

setup(name="LoProp",
    version="0.1",
    packages=[".", "test", "daltools", "daltools/util"],
    scripts=["loprop.py",],
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    )

    
