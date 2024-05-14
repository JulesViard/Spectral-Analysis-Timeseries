import os.path as osp
import re
import sys
from setuptools import setup, find_packages



def get_script_path():
    return osp.dirname(osp.realpath(sys.argv[0]))


def read(*parts):
    return open(osp.join(get_script_path(), *parts)).read()


def find_version(*parts):
    vers_file = read(*parts)
    match = re.search(r'^__version__ = "(\d+\.\d+\.\d+)"', vers_file, re.M)
    if match is not None:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="a_ssa",
    version=find_version('src', 'a_ssa', '__init__.py'),
    packages=find_packages('src'),
    package_dir={'': 'src'},
    author="Jules Viard",
    author_email="jules.viard@xfel.eu",
    description="Singular Spectral Analysis package for Time Series Analysis",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://git.xfel.eu/viardj/singular-spectral-analys",
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
        'river',
        'pwlf'
    ],
    extras_require={
        'test': [
            'pytest',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
)
