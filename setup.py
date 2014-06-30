#!/usr/bin/env python

from distutils.core import setup

setup(
    name='gptools',
    version='0.1',
    packages=['gptools', 'gptools.kernel'],
    requires=['scipy', 'numpy', 'matplotlib', 'mpmath', 'emcee', 'triangle'],
    author='Mark Chilenski',
    author_email='mark.chilenski@gmail.com',
    url='https://github.com/markchil/gptools',
    description='Gaussian process regression with derivative constraints and predictions.',
    long_description=open('README.rst', 'r').read(),
    license='GPL'
)