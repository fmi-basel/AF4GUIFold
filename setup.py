import setuptools

setuptools.setup(
     name='guifold_alphafold',
     version='0.4',
     zip_safe=False,
     packages=setuptools.find_packages(),
     include_package_data=True,
     scripts=['run_alphafold.py'],)
