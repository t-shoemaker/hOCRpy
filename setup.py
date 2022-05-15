#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'hOCR parser and analysis package'
LONG_DESCRIPTION = 'Parse and analyze structured OCR output'

setup(
    name='hOCRpy',
    version=VERSION,
    author='Tyler Shoemaker',
    author_email='tshoemaker@ucdavis.edu',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[]
)
