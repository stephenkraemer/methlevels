from setuptools import setup, find_packages

setup(name='methlevels',
      version='0.1',
      description='methlevels',
      long_description='methlevels',
      author='Stephen Kraemer',
      author_email='stephenkraemer@gmail.com',
      license='MIT',

      packages=find_packages(where='src'),
      package_dir={'': 'src'},

      install_requires=[
          'pandas>=0.23.4',
          'joblib>=0.12',
          'matplotlib>=3.0.0',
          'seaborn>=0.9.0',
          'numpy',
          'pyranges',
      ],

      # also requires bedtools and htslib, cf. meta.yaml in conda recipe

      extras_require={
          'dev': [
              'pytest',
              'mypy',
          ],
      },

      dependency_links=[
          'git+https://github.com/numpy/numpy-stubs.git#egg=numpy-stubs-0.01',
      ],

)
