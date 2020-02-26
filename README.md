[![CircleCI](https://circleci.com/gh/vahtras/loprop.svg?style=svg)](https://circleci.com/gh/vahtras/loprop)
[![Coverage Status](https://coveralls.io/repos/github/vahtras/loprop/badge.svg?branch=master)](https://coveralls.io/github/vahtras/loprop?branch=master)
![Python application](https://github.com/vahtras/loprop/workflows/Python%20application/badge.svg?event=push)

# Loprop for Dalton


This code is an implementation of the LoProp algorithm based on Gagliardi et al., JCP **121**, 4494 (2004) for postprocessing calculation with Dalton (http://daltonprogram.org)

## How to cite this software

LoProp for Dalton, Olav Vahtras (2014). http://dx.doi.org/10.5281/zenodo.13276

## Requirements

A python installation with `numpy` and `scipy` libraries

## Installation

### pypi (recommended)

```bash
$ pip install loprop
```

### github

```bash
$ git clone https://github.com/vahtras/loprop.git
$ cd loprop
$ pip install -r requirements.txt
```

This will automatically put loprop into your path.  
Works with virtualenv and anaconda.


## Test

With `nose` installed one can travers all tests which should give

```
$ pip install nose
$ python -m nose
.......................................................................................................................................................................................................................................................................
----------------------------------------------------------------------
Ran 263 tests in 12.499s

OK
```

## Basic usage

To setup a Dalton calculation for postprocessing with loprop, a typical input file is as follows

```
**DALTON INPUT
.RUN RESP
*END OF GENERAL
**WAVE FUNCTION
.INTERFACE
.HF
**INTEGRAL
.NOSUP
.DIPLEN
.SECMOM
**RESPONSE
*LINEAR
.DIPLEN
*END OF
```

This is required for calculating atomic dipoles, quadrupoles and polarizabilities
One-electron integral files are required that are not normally saved after a Dalton calculation. The dalton program should be executed with the following options

```
$ dalton -get "AOONEINT AOPROPER" hf h2o
```

A sample run with charges and isotropic polarizabilities is
```
$ loprop -f hf_h2o.tar.gz -l 0 -a 1
AU
3 0 1 1
1     0.000     0.000     0.698    -0.703     3.466
1    -1.481     0.000    -0.349     0.352     1.576
1     1.481     0.000    -0.349     0.352     1.576
```
generating a potential file, with local coordinates, charge and polarizability for each atom.

