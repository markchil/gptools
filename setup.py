#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='gptools',
    version='0.1.2',
    packages=['gptools', 'gptools.kernel'],
    install_requires=['scipy', 'numpy', 'matplotlib', 'mpmath', 'emcee', 'triangle_plot'],
    author='Mark Chilenski',
    author_email='mark.chilenski@gmail.com',
    url='https://github.com/markchil/gptools',
    description='Gaussian process regression with derivative constraints and predictions.',
    long_description=open('README.rst', 'r').read(),
    license='GPL'
)