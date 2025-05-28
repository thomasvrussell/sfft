Roman SNPIT fork of SFFT
------------------------

Please look at the PyPi package ``sfft`` for the original.  This
package has customizations created for the Roman SNPIT.

..  image:: https://github.com/thomasvrussell/sfft/blob/master/docs/sfft_logo_gwbkg.png

*SFFT: Saccadic Fast Fourier Transform for image subtraction*
---------------------
.. image:: https://img.shields.io/pypi/v/sfft.svg
    :target: https://pypi.python.org/pypi/sfft
    :alt: Latest Version

.. image:: https://static.pepy.tech/personalized-badge/sfft?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
    :target: https://pepy.tech/project/sfft

.. image:: https://img.shields.io/badge/python-3.7-green.svg
    :target: https://www.python.org/downloads/release/python-370/

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.6463000.svg
    :target: https://doi.org/10.5281/zenodo.6463000
    :alt: 1.0.6

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
|
Saccadic Fast Fourier Transform (SFFT) is an algorithm for fast & accurate image subtraction in Fourier space. 
SFFT brings about a remarkable improvement of computational performance of around an order of magnitude compared to other published image subtraction codes. 

SFFT method is the transient detection engine for several ongoing time-domain programs, including the `DESIRT <https://ui.adsabs.harvard.edu/abs/2022TNSAN.107....1P/abstract>`_ survey based on DECam & DESI, the DECam GW-MMADS Survey for GW Follow-ups and the JWST Cycle 3 Archival program `AR 5965 <https://www.stsci.edu/jwst/science-execution/program-information?id=5965>`_. SFFT is also the core engine for the differential photometry pipeline of the `Roman Supernova PIT <https://github.com/Roman-Supernova-PIT>`_.

Get started
---------------------

- **Documentation:** https://thomasvrussell.github.io/sfft-doc/ **[recommended]**
- **Installation:** https://thomasvrussell.github.io/sfft-doc/installation/
- **Tutorials:** https://thomasvrussell.github.io/sfft-doc/tutorials/
- **Source code:** https://github.com/thomasvrussell/sfft
- **Contact the author:** astroleihu@gmail.com or leihu@andrew.cmu.edu

Installation
=================
To install the latest release from PyPI, use pip: ::
    
    pip install sfft

For more detailed instructions, see the `install guide <https://thomasvrussell.github.io/sfft-doc/installation/>`_ in the docs.

Citing
--------

*Image Subtraction in Fourier Space, Lei Hu et al. 2022, The Astrophysical Journal, 936, 157* 

See ADS Link: https://ui.adsabs.harvard.edu/abs/2022ApJ...936..157H/abstract

Publications using SFFT method
--------------------------------

See ADS Library: https://ui.adsabs.harvard.edu/public-libraries/lc4tiTR_T--92f9k0YrRQg


Roman SNPIT Variant Notes
-------------------------

(These notes are for members of the Roman SNPIT who need to upload a new
version of this package.)

* Make sure that ``NAME`` in ``setup.py`` is ``sfft-romansnpit``.

* Make sure that VERSION in ``setup.py`` is of the form
``<MAJOR>.<MINOR>.<PATCH>.dev<n>`` where ``<MAJOR>``,
``<MINOR>``, AND ``<PATCH>`` match the upstream that has most recently been
merged in, and ``<n>`` is an integer that can start at 0 for a given
``MAJOR.MINOR.PATCH`` and that needs to increment each time we want to
push a new package.

* Build in the main directory with::

    python -m build --sdist --outdir dist

  That will create a file in ``dist`` named
  ``sfft_romansnpit-<version>.tar.gz`` where ``<version>`` is what you
  put in the ``VERSION`` variable.

* You can then upload it to the ``sfft-romansnpit`` package on PyPi with::

    twine upload dist/sfft_romansnpit-<version>.tar.gz
