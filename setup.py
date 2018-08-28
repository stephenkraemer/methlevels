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
          'pandas',
          'numpy'
      ],

      extras_require={
          'dev': [
              'pytest',
              'pytest-mock',
              'pytest-xdist',
              'mypy==0.610',
              'numpy-stubs',
          ],
      },

      dependency_links=[
          'git+https://github.com/numpy/numpy-stubs.git#egg=numpy-stubs-0.01',
      ],

)
