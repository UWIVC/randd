Rate and Distortion: A toolkit for video codec comparison
=========================================================

.. intro-section-start

R&D is a Python library that faciliates multimedia codec comparison. It provides functions to perform rate-distortion analysis for a given codec with sparse samples in the rate-distortion space.

.. intro-section-end

.. feature-section-start
========
Features
========

* Efficiency: R&D provides the implementation of state-of-the-art rate-distortion function models, which enables efficient codec comparison with only sparse encoding samples.
* Accuracy: R&D is demonstrated to be more accuracy than the traditional BD-rate measure.
* Flexibility: R&D can be applied to analyze the encoder performance at different resolution, frame rate, and bit depth, with a unified interface.
* Robustness: R&D curve/surface fitting models are mathematically stable.
* Generalizability: R&D is compatible with most of the modern video quality assessment models such as PSNR, VMAF, and SSIMPlus.
* Extrapolation Capability: R&D can deliver reasonable codec performance analysis in the region where no encoding samples are provided.

.. feature-section-end

.. installation-section-start

`R&D <https://github.com/UWIVC/randd>_` can be installed using ``pip`` or from source.

Install from source:

.. code-block:: sh
    git clone git@github.com:UWIVC/randd.git
    python setup.py install

Install from pip:

.. code-block:: sh
    pip install randd

.. installation-section-end


.. usage-example-start
1D example with distortion measure being PSNR:

.. code-block:: python
    import randd as rd
    import numpy as np

    r1 = np.array([100, 300, 800, 1500])
    d1 = np.array([])
    r2 = np.array([100, 300, 800, 1500])
    d2 = np.array([])
    analyzer = rd.Analyzer(d_measure='psnr')
    quality_gain, bitrate_saving, summary = analyzer(r1, d1, r2, d2, codec1='h264', codec2='hevc')


2D example with distortion measure being VMAF. Compare the two codecs in the bitrate region [100, 3000]:

.. code-block:: python
    import randd as rd
    import numpy as np

    r1 = np.array([100, 300, 800, 1500])
    d1 = np.array([])
    r2 = np.array([100, 300, 800, 1500])
    d2 = np.array([])
    analyzer = rd.Analyzer(d_measure='vmaf', ndim=2, r_roi=[100, 3000])
    quality_gain, bitrate_saving, summary = analyzer(r1, d1, r2, d2, codec1='h264', codec2='hevc')


.. usage-example-end


.. contact-section-start

Contacts
--------

**Wentao Liu** - `@w238liu <https://ece.uwaterloo.ca/~w238liu>`_ - ``w238liu@uwaterloo.ca``
**Zhengfang Duanmu** - `@zduanmu <https://ece.uwaterloo.ca/~zduanmu>`_ - ``zduanmu@uwaterloo.ca``

.. contact-section-end