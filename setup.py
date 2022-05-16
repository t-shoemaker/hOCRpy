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
    install_requires=[
        'lxml==4.8.0',
        'matplotlib==3.5.2',
        'numpy==1.21.6',
        'Pillow==9.1.0',
        'scipy==1.7.3', 
    ]
)
