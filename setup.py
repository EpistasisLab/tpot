#!/usr/bin/env python
# -*- coding: utf-8 -*-
#TODO update this
from setuptools import setup, find_packages


def calculate_version():
    initpy = open('tpot2/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version


package_version = calculate_version()

setup(
    name='TPOT2',
    python_requires='<3.12', #for configspace compatibility
    version=package_version,
    author='Pedro Ribeiro',
    packages=find_packages(),
    url='https://github.com/EpistasisLab/tpot2',
    license='GNU/LGPLv3', #TODO
    entry_points={'console_scripts': ['tpot2=tpot2:main', ]},
    description=('Tree-based Pipeline Optimization Tool'),
    long_description='''
A Python tool that automatically creates and optimizes machine learning pipelines using genetic programming.


''',
    zip_safe=True,
    install_requires=['numpy==1.26.4',
                      'scipy>=1.3.1',
                      'scikit-learn>=1.3.0',
                      'update_checker>=0.16',
                      'tqdm>=4.36.1',
                      'stopit>=1.1.1',
                      'pandas>=2.2.0',
                      'joblib>=1.1.1',
                      'xgboost>=1.7.0',
                      'matplotlib>=3.6.2',
                      'traitlets>=5.8.0',
                      'lightgbm>=3.3.3',
                      'optuna>=3.0.5',
                      'baikal>=0.4.2',
                      'networkx>=3.0',
                      'dask>=2024.4.2',
                      'distributed>=2024.4.2',
                      'dask-expr>=1.0.12',
                      'dask-jobqueue>=0.8.5',
                      'func_timeout>=4.3.5',
                      'configspace>=0.7.1',
                     ],
    extras_require={
        'skrebate': ['skrebate>=0.3.4'],
        'mdr': ['scikit-mdr>=0.4.4'],
        'sklearnex' : ['scikit-learn-intelex>=2023.2.1'],
        'amltk' : ['amltk>=1.12.1'],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=['pipeline optimization', 'hyperparameter optimization', 'data science', 'machine learning', 'genetic programming', 'evolutionary computation'],
)
