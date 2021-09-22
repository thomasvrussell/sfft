import os
import sys
from setuptools import setup, find_packages

DESCRIPTION = "Image Subtraction in Fourier Space"
LONG_DESCRIPTION = open('README.rst').read()
NAME = "sfft"
AUTHOR = "Lei Hu"
AUTHOR_EMAIL = "hulei@pmo.ac.cn"
MAINTAINER = "Lei Hu"
MAINTAINER_EMAIL = "hulei@pmo.ac.cn"
DOWNLOAD_URL = 'https://github.com/thomasvrussell/sfft'
LICENSE = 'MIT Licence'
VERSION = '1.0.3'

install_reqs = ['scipy>=1.5.2',
                'astropy>=3.2.3',
                'scikit-image>=0.16.2',
                'sympy>=1.4',
                'fastremap>=1.7.0',
                'sep>=1.0.3',
                'numba==0.53.1',
                'llvmlite==0.36.0',
                'pyfftw==0.12.0']

setup(name = NAME,
      version = VERSION,
      description = DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      setup_requires = ['numpy'],
      install_requires = install_reqs,
      author = AUTHOR,
      author_email = AUTHOR_EMAIL,
      maintainer = MAINTAINER,
      maintainer_email = MAINTAINER_EMAIL,
      download_url = DOWNLOAD_URL,
      license = LICENSE,
      packages = find_packages(),
      include_package_data = True,
      classifiers = [
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Astronomy'],
     )
