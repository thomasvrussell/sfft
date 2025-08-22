from setuptools import setup, find_packages

DESCRIPTION = "Image Subtraction in Fourier Space"
LONG_DESCRIPTION = open('README.rst').read()
NAME = "sfft-romansnpit"
AUTHOR = "Lei Hu"
AUTHOR_EMAIL = "leihu@andrew.cmu.edu"
MAINTAINER = "Lei Hu"
MAINTAINER_EMAIL = "leihu@andrew.cmu.edu"
DOWNLOAD_URL = 'https://github.com/thomasvrussell/sfft'

LICENSE = 'MIT License'
VERSION = '1.6.4.dev10'

install_reqs = ['scipy>=1.5.2',
                'astropy>=3.2.3',
                'scikit-image>=0.25.2',
                'fastremap>=1.7.0',
                'sep>=1.0.3',
                'numba>=0.53.1',
                'llvmlite>=0.36.0',
                'pyfftw>=0.12.0']

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      setup_requires=['numpy'],
      install_requires=install_reqs,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Astronomy'],
     )
