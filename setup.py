import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension
from multiprocessing import set_start_method
try:
    set_start_method("fork")
except RuntimeError:
    pass
from multiprocessing import cpu_count

DESCRIPTION = "Image Subtraction in Fourier Space"
LONG_DESCRIPTION = open('README.rst').read()
NAME = "sfft"
AUTHOR = "Lei Hu"
AUTHOR_EMAIL = "leihu@sas.upenn.edu"
MAINTAINER = "Lei Hu"
MAINTAINER_EMAIL = "leihu@sas.upenn.edu"
DOWNLOAD_URL = 'https://github.com/thomasvrussell/sfft'

LICENSE = 'MIT Licence'
VERSION = '1.7.1'

install_reqs = ['scipy>=1.5.2',
                'astropy>=3.2.3',
                'fastremap>=1.7.0',
                'sep>=1.0.3',
                'numba>=0.53.1',
                'llvmlite>=0.36.0',
                'pyfftw>=0.12.0']

# Cythonize the extensions
cythonized_extensions = cythonize([
    Extension(
        "sfft.utils.houghLine._hough_transform",
        ["sfft/utils/houghLine/_hough_transform.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "sfft.utils.houghLine._ccomp",
        ["sfft/utils/houghLine/_ccomp.pyx"],
        include_dirs=[numpy.get_include()]
    )
], nthreads=cpu_count(), compiler_directives={'language_level': 3})

if __name__ == "__main__":
    setup(
        name=NAME,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        # setup_requires=['numpy'],
        install_requires=install_reqs,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        python_requires='>=3.5',
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        ext_modules=cythonized_extensions,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'Natural Language :: English',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3 :: Only',
            'Topic :: Scientific/Engineering :: Astronomy']
    )
