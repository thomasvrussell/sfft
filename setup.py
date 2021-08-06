import os
import sys
from setuptools import setup, find_packages

DESCRIPTION = "Image Subtraction in Fourier Space"
LONG_DESCRIPTION = open('README.md').read()
NAME = "sfft"
AUTHOR = "Lei Hu"
AUTHOR_EMAIL = "hulei@pmo.ac.cn"
MAINTAINER = "Lei Hu"
MAINTAINER_EMAIL = "hulei@pmo.ac.cn"
DOWNLOAD_URL = 'https://github.com/thomasvrussell/sfft'
LICENSE = 'MIT Licence'
VERSION = '1.0.dev0'

if "--CUDA_VERSION" not in sys.argv:
    sys.exit("ERROR: Please specify --CUDA_VERSION {'NONE', '10.1', '11.2'}) !")

index = sys.argv.index('--CUDA_VERSION')
sys.argv.pop(index)                     # Removes the '--CUDA_VERSION'
CUDA_VERSION = sys.argv.pop(index)      # Returns the element after the '--CUDA_VERSION'
assert CUDA_VERSION in ['NONE', '10.1', '11.2']   # NOTE: find your CUDA version by terminal command 'nvidia-smi'

install_reqs = ['scipy>=1.5.2',
                'astropy>=3.2.3',
                'scikit-image>=0.16.2',
                'fastremap>=1.7.0',
                'sep>=1.0.3']

install_reqs += ['numba==0.53.1',
                 'pyfftw==0.12.0']

print('\n********************************************** sfft **************************************************\n')
if CUDA_VERSION == 'NONE':
    print('*** IMPORTANT NOTICE: A pure CPU version of sfft will be installed !')
    pass

if CUDA_VERSION == '10.1':
    print('*** WARNING: Please make sure you have installed cudatoolkit=10.1 (e.g., via conda) !')
    print('*** IMPORTANT NOTICE: CPU & GPU (pycuda) & GPU (cupy) versions of sfft will be installed !')
    
    install_reqs += ['pycuda==2020.1',
                     'scikit-cuda==0.5.3',
                     'cupy-cuda101>=8.5.0']

if CUDA_VERSION == '11.2':
    print('*** WARNING: Please make sure you have installed cudatoolkit=11.2 (e.g., via conda) !')
    print('*** WARNING: GPU (pycuda) version is currently not available for CUDA 11 !')
    print('*** IMPORTANT NOTICE: CPU & GPU (cupy) versions of sfft will be installed !')

    install_reqs += ['cupy-cuda112>=9.0.0']
print('\n********************************************** sfft **************************************************\n')

setup(name = NAME,
      version = VERSION,
      description = DESCRIPTION,
      long_description = LONG_DESCRIPTION,
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
        'Development Status :: Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Astronomy'],
     )
