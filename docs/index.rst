.. U-FISH documentation master file

Welcome to U-FISH Documentation
================================

.. image:: https://img.shields.io/pypi/v/ufish.svg
   :target: https://pypi.org/project/ufish/
   :alt: PyPI version

U-FISH 🎣 is an advanced FISH spot calling algorithm based on deep learning. The "U" in U-FISH represents both the U-Net architecture and the Unified output of enhanced images, underpinning our design philosophy.

.. image:: ufish.png
   :alt: U-FISH Overview
   :align: center

Key Features
------------

* **Diverse dataset**: 4000+ images with approximately 1.6 million targets from seven sources
* **Small model**: State-of-the-art performance with only 160k parameters (ONNX file size: 680kB)
* **3D support**: Detection of FISH spots in 3D images
* **Scalability**: Support for large-scale data storage formats (OME-Zarr and N5)
* **User-friendly interface**: API, CLI, Napari plugin, and web application

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install ufish

For GPU inference support:

.. code-block:: bash

   pip install onnxruntime-gpu

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from skimage import io
   from ufish.api import UFish

   # Initialize U-FISH
   ufish = UFish()
   ufish.load_weights()

   # Predict spots
   img = io.imread("path/to/image.tiff")
   pred_spots, enh_img = ufish.predict(img)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   cli_reference
   tutorials
   benchmarks
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 