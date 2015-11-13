#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


def parse_requirements(filename):
    return list(filter(lambda line: (line.strip())[0] != '#',
                       [line.strip() for line in open(filename).readlines()]))


def calculate_version():
    initpy = open('tpot/__init__.py').read().split('\n')
    version = next(filter(lambda x: '__version__' in x, initpy)).split('\'')[1]
    return version


requirements = parse_requirements('requirements.txt')
package_version = calculate_version()

setup(
    name='TPOT',
    version=package_version,
    author='Randal S. Olson',
    author_email='rso@randalolson.com',
    packages=find_packages(),
    url='https://github.com/rhiever/tpot',
    license='GNU/GPLv3',
    entry_points={"console_scripts": ["tpot=tpot:main", ]},
    description=('Tree-based Pipeline Optimization Tool'),
    long_description='''
A Python tool that automatically creates and optimizes Machine Learning pipelines using genetic programming.

Contact
=============
If you have any questions or comments about TPOT, please feel free to contact me via:

E-mail: rso@randalolson.com

or Twitter: https://twitter.com/randal_olson

This project is hosted at https://github.com/rhiever/tpot
''',
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
