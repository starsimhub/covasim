'''
Covasim installation. Requirements are listed in requirements.txt. There are two
options:
    pip install -e .       # Standard install, does not include optional libraries
    pip install -e .[full] # Full install, including optional libraries
'''

import os
import sys
import runpy
from setuptools import setup, find_packages

# Load requirements from txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'covasim', 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Get the documentation
with open(os.path.join(cwd, 'README.rst'), "r") as f:
    long_description = f.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]

setup(
    name="covasim",
    version=version,
    author="Cliff Kerr, Robyn Stuart, Romesh Abeysuriya, Dina Mistry, Lauren George, and Daniel Klein, on behalf of the IDM COVID-19 Response Team",
    author_email="info@covasim.org",
    description="COVID-19 Agent-based Simulator",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='http://covasim.org',
    keywords=["COVID", "COVID-19", "coronavirus", "SARS-CoV-2", "stochastic", "agent-based model", "interventions", "epidemiology"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'full':  [
            'plotly',
            'fire',
            'optuna',
            'synthpops',
            'parestlib',
        ],
    }
)

