#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


def calculate_version():
    initpy = open('tpot/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version


package_version = calculate_version()

setup(
    name='TPOT',
    version=package_version,
    author='Randal S. Olson',
    author_email='rso@randalolson.com',
    packages=find_packages(),
    url='https://github.com/EpistasisLab/tpot',
    license='GNU/LGPLv3',
    entry_points={'console_scripts': ['tpot=tpot:main', ]},
    description=('Tree-based Pipeline Optimization Tool'),
    long_description='''
A Python tool that automatically creates and optimizes machine learning pipelines using genetic programming.

Contact
=============
If you have any questions or comments about TPOT, please feel free to contact us via:

E-mail: ttle@pennmedicine.upenn.edu or weixuanf@pennmedicine.upenn.edu

or Twitter: https://twitter.com/trang1618 or https://twitter.com/WeixuanFu

This project is hosted at https://github.com/EpistasisLab/tpot
''',
    zip_safe=True,
    install_requires=['numpy>=1.16.3',
                    'scipy>=1.3.1',
                    'scikit-learn>=0.22.0',
                    'deap>=1.2',
                    'update_checker>=0.16',
                    'tqdm>=4.36.1',
                    'stopit>=1.1.1',
                    'pandas>=0.24.2',
                    'joblib>=0.13.2'],
    extras_require={
        'xgboost': ['xgboost==0.90'],
        'skrebate': ['skrebate>=0.3.4'],
        'mdr': ['scikit-mdr>=0.4.4'],
        'dask': ['dask>=0.18.2',
                 'distributed>=1.22.1',
                 'dask-ml>=1.0.0'],
        'torch': ['torch==1.3.1'],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=['pipeline optimization', 'hyperparameter optimization', 'data science', 'machine learning', 'genetic programming', 'evolutionary computation'],
)
