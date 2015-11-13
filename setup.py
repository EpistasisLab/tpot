#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os
import subprocess


def parse_requirements(filename):
    return list(filter(lambda line: (line.strip())[0] != '#',
                       [line.strip() for line in open(filename).readlines()]))


def calculate_version():
    initpy = open('tpot/__init__.py').read().split('\n')
    version = next(filter(lambda x: '__version__' in x, initpy)).split('\'')[1]
    return version


requirements = parse_requirements('requirements.txt')
version_git = calculate_version()


def get_long_description():
    readme_file = 'README.md'
    if not os.path.isfile(readme_file):
        print('warning: README.md not found')
        return ''
    # Try to transform the README from Markdown to reStructuredText.
    try:
        from pypandoc import convert
        read_md = convert(readme_file, 'rst')
    except ImportError:
        print('warning: pypandoc module not found, could not convert Markdown to RST')
        read_md = open(readme_file, 'r').read()
    return read_md

setup(
    name='TPOT',
    version=version_git,
    author='Randal S. Olson',
    author_email='rso@randalolson.com',
    packages=find_packages(),
    url='https://github.com/rhiever/tpot',
    license='GNU/GPLv3',
    description=('A Python tool that automatically creates and optimizes Machine '
                 'Learning pipelines using genetic programming.'),
    #long_description=get_long_description(),
    zip_safe=True,
    install_requires=requirements,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords=['pipeline optimization', 'hyperparameter optimization', 'data science', 'machine learning', 'genetic programming', 'evolutionary computation'],
)
