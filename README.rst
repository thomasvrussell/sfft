..  image:: https://github.com/thomasvrussell/sfft/blob/master/docs/sfft_logo_gwbkg.png

SFFT
---------------------
*Python package for Saccadic Fast Fourier Transform (SFFT) image subtraction*

.. image:: https://img.shields.io/pypi/v/sfft.svg
    :target: https://pypi.python.org/pypi/sfft
    :alt: Latest Version

.. image:: https://static.pepy.tech/personalized-badge/sfft?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
    :target: https://pepy.tech/project/sfft

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.6463000.svg
    :target: https://doi.org/10.5281/zenodo.6463000
    :alt: 1.0.6

.. image:: https://img.shields.io/badge/python-3.7-green.svg
    :target: https://www.python.org/downloads/release/python-370/

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/licenses/MIT

Saccadic Fast Fourier Transform (SFFT) is an algorithm for image subtraction in Fourier space. SFFT brings about a remarkable improvement of computational performance of around an order of magnitude compared to other published image subtraction codes. 

SFFT method is the transient detection engine for several ongoing time-domain programs, including the `DESIRT <https://ui.adsabs.harvard.edu/abs/2022TNSAN.107....1P/abstract>`_ survey based on DECam & DESI, the DECam GW-MMADS Survey for GW Follow-ups and the JWST Cycle 3 Archival program `AR 5965 <https://www.stsci.edu/jwst/science-execution/program-information?id=5965>`_. SFFT is also the core engine for the differential photometry pipeline of the `Roman Supernova PIT <https://github.com/Roman-Supernova-PIT>`_.

- **Home Page:** https://thomasvrussell.github.io/sfft-doc/
- **Installation:** https://thomasvrussell.github.io/sfft-doc/installation/
- **Tutorials:** https://thomasvrussell.github.io/sfft-doc/tutorials/

Citing
--------

*Image Subtraction in Fourier Space. Lei Hu et al. 2022, The Astrophysical Journal, 936, 157*

Arxiv link: `<https://arxiv.org/abs/2109.09334>`_.

ApJ Publication link: `<https://doi.org/10.3847/1538-4357/ac7394>`_.

Related DOI: 10.3847/1538-4357/ac7394

Publications using SFFT method
--------------------------------

See ADS Library: https://ui.adsabs.harvard.edu/public-libraries/lc4tiTR_T--92f9k0YrRQg

Acknowledgment
--------------------------------

The version SFFT v1.6.* is developed during the `NASA GPU Hackathon 2024 <https://www.nas.nasa.gov/hackathon/#home>`_ as part of the pipeline optimization efforts for the `Roman Supernova PIT team <https://github.com/Roman-Supernova-PIT>`_. The valuable support and insightful suggestions from our team members and Hackathon mentors are essential for the improvements in this version of SFFT. 

I would like to specifically acknowledge the contributions of the following team members: Lauren Aldoroty (Duke), Robert Knop (LBNL), Shu Liu (Pitt), and Michael Wood-Vasey (Pitt). Additionally, I am grateful for the guidance provided by our mentors, Marcus Manos and Lucas Erlandson from NVIDIA.
