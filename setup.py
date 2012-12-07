#!/usr/bin/env python

from distutils.core import setup

setup(name='ocl',
      version='0.7',
      description='Decorators to compile Python code to C99, OpenCL, and JS',
      author='Massimo Di Pierro',
      author_email='massimo.dipierro@gmail.com',
      license='bsd',
      url='https://github.com/mdipierro/ocl',
      py_modules = ['ocl'],
      requires=['meta (>=0.4)']
      )

