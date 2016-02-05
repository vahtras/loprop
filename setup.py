#!/usr/bin/env python

try:
    from setuptools import setup, find_packages
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
    packages = find_packages(),
    package_data = { 'loprop.test' : [ '*/tmp/RSPVEC',
                                       '*/tmp/DALTON.BAS',
                                       '*/tmp/AOONEINT',
                                       '*/tmp/SIRIFC',
                                       '*/tmp/AOPROPER', ],
                     'loprop.daltools.test' : ['data/*', 
                         'test*/RSPVEC',
                         'test*/DALTON.BAS',
                         'test*/AOONEINT',
                         'test*/SIRIFC',
                         'test*/AOPROPER',
                         'test*/E3VEC',
                         ],
                     'loprop.daltools.util.test' : [ 'fort.1', 'fort.2', 'fort.3'], 
                     },
    scripts=["loprop/loprop.py",],
    author="Olav Vahtras",
    author_email="vahtras@kth.se",
    install_requires = reqs,
    description = 'LoProp implementation for Dalton',
    )
