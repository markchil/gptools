#!/usr/bin/env python

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import numpy
from Cython.Distutils import build_ext

_matern = Extension(
    "gptools.kernel._matern",
    ["gptools/kernel/_matern.pyx", "gptools/kernel/src/matern.c"],
    include_dirs=[numpy.get_include(), 'gptools/kernel/include']
)

setup(
    name='gptools',
    version='0.2.2',
    packages=['gptools', 'gptools.kernel'],
    install_requires=['scipy', 'numpy', 'matplotlib', 'mpmath', 'emcee', 'triangle_plot'],
    author='Mark Chilenski',
    author_email='mark.chilenski@gmail.com',
    url='https://github.com/markchil/gptools',
    description='Gaussian process regression with derivative constraints and predictions.',
    long_description=open('README.rst', 'r').read(),
    cmdclass={'build_ext': build_ext},
    ext_modules = [_matern],
    license='GPL',
    headers=['gptools/kernel/include/matern.h']
)