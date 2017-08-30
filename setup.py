from distutils.core import setup
from setuptools import find_packages


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='emb4class',
      version='0.1',
      description='Word embedding benchmark for short text classification',
      long_description = readme(),
      url='http://github.com/PauBatlle/emb4class',
      author='Pau Batlle',
      author_email='pau.batlle@estudiant.upc.edu',
      license='MIT',
      install_requires = ["numpy", "tensorflow", "matplotlib", "seaborn", "scikit-learn", "nltk"],
      packages=find_packages(),
      python_requires = '>=3.3',
      classifiers=['Development Status :: 3 - Alpha'])
