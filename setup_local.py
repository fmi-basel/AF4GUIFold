import setuptools
from setuptools import find_packages

setuptools.setup(
    name='alphafold4guifold',
    version='2.2.0',
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    scripts=['alphafold/run_alphafold.py',],)
