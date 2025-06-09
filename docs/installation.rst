Installation Guide
==================

This guide will help you install U-FISH on your system.

Requirements
------------

* Python 3.9 or higher
* pip package manager

Basic Installation
------------------

The simplest way to install U-FISH is via pip:

.. code-block:: bash

   pip install ufish

This will install U-FISH with CPU support for inference.

GPU Support
-----------

For Inference
~~~~~~~~~~~~~

To enable GPU acceleration for inference, install the GPU version of ONNX Runtime:

.. code-block:: bash

   pip install onnxruntime-gpu

For Training
~~~~~~~~~~~~

If you plan to train or fine-tune models, you'll need PyTorch with CUDA support:

.. code-block:: bash

   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Visit the `PyTorch official website <https://pytorch.org/>`_ for more installation options.

Windows with AMD GPU
~~~~~~~~~~~~~~~~~~~~

If you're using Windows with an AMD GPU, install the DirectML backend:

.. code-block:: bash

   pip install torch-directml

Development Installation
------------------------

To install U-FISH for development:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/GangCaoLab/U-FISH.git
      cd U-FISH

2. Install in development mode:

   .. code-block:: bash

      pip install -e .

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

Optional Dependencies
---------------------

Napari Plugin
~~~~~~~~~~~~~

To use the Napari plugin for visualization:

.. code-block:: bash

   pip install napari[all]
   pip install napari-ufish

Large-scale Data Support
~~~~~~~~~~~~~~~~~~~~~~~~

For working with large-scale data formats:

.. code-block:: bash

   # For Zarr support
   pip install zarr

   # For OME-Zarr support
   pip install ome-zarr

   # For N5 support
   pip install pyn5

Verifying Installation
----------------------

To verify that U-FISH is installed correctly:

.. code-block:: python

   import ufish
   print(ufish.__version__)

You can also check the CLI:

.. code-block:: bash

   ufish --help

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **ImportError: No module named 'ufish'**
   
   Make sure you've activated the correct Python environment where U-FISH is installed.

2. **CUDA out of memory errors**
   
   Try reducing the batch size or image size, or use CPU inference instead.

3. **ONNX Runtime errors**
   
   Ensure you have the correct version of onnxruntime or onnxruntime-gpu installed.

Getting Help
~~~~~~~~~~~~

If you encounter any issues:

1. Check the `GitHub issues <https://github.com/GangCaoLab/U-FISH/issues>`_
2. Create a new issue with detailed information about your problem
3. Include your system information (OS, Python version, GPU model if applicable) 