Changelog
=========

All notable changes to U-FISH will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
* Comprehensive Read the Docs documentation
* Support for multi-GPU training
* Batch processing optimizations
* New tutorial notebooks

Changed
~~~~~~~
* Improved memory efficiency for large images
* Enhanced 3D processing performance

Fixed
~~~~~
* Memory leak in tiling mode
* Edge case in spot matching algorithm

[1.0.0] - 2024-01-15
--------------------

This is the first stable release of U-FISH.

Added
~~~~~
* Core FISH spot detection functionality
* U-Net based image enhancement
* Support for 2D and 3D images
* Pre-trained models for various FISH types
* Python API for programmatic access
* Command-line interface
* Evaluation metrics and visualization tools
* Support for large-scale data formats (Zarr, N5, OME-Zarr)
* GPU acceleration with ONNX Runtime
* Comprehensive test suite
* Example notebooks and tutorials

Features
~~~~~~~~
* **High accuracy**: State-of-the-art F1 scores on benchmark datasets
* **Small model**: Only 160k parameters (680KB ONNX file)
* **Fast inference**: Real-time processing on GPU
* **Flexible**: Works with diverse FISH imaging protocols
* **User-friendly**: Simple API and CLI

[0.9.0] - 2023-12-01
--------------------

Beta release for testing and feedback.

Added
~~~~~
* Training and fine-tuning capabilities
* Data augmentation pipeline
* Mixed precision training support
* Early stopping and learning rate scheduling
* Model checkpointing
* TensorBoard integration

Changed
~~~~~~~
* Improved model architecture for better accuracy
* Optimized data loading pipeline
* Enhanced CLI with more options

Fixed
~~~~~
* Numerical stability in loss calculation
* Memory issues with large batch sizes

[0.8.0] - 2023-10-15
--------------------

Alpha release with core functionality.

Added
~~~~~
* Basic U-Net implementation
* Simple spot detection algorithm
* Initial API design
* Basic CLI commands
* Preliminary documentation

Changed
~~~~~~~
* Switched from TensorFlow to PyTorch
* Adopted ONNX for model deployment

[0.7.0] - 2023-08-01
--------------------

Pre-alpha development version.

Added
~~~~~
* Project structure
* Initial U-Net experiments
* Dataset collection and curation
* Benchmark framework

Migration Guides
----------------

Migrating from 0.x to 1.0
~~~~~~~~~~~~~~~~~~~~~~~~~

The 1.0 release includes several API changes:

1. **Model Loading**:

   .. code-block:: python

      # Old (0.x)
      model = UFish.load_model('path/to/model')
      
      # New (1.0)
      ufish = UFish()
      ufish.load_weights('path/to/model')

2. **Prediction API**:

   .. code-block:: python

      # Old (0.x)
      spots = model.predict(image)
      
      # New (1.0)
      spots, enhanced = ufish.predict(image)

3. **CLI Commands**:

   .. code-block:: bash

      # Old (0.x)
      ufish-predict input.tiff output.csv
      
      # New (1.0)
      ufish predict input.tiff output.csv

Deprecation Policy
------------------

* Features marked as deprecated will be maintained for at least 2 minor versions
* Deprecation warnings will be issued when using deprecated features
* Migration guides will be provided for all breaking changes

Future Releases
---------------

Planned features for future releases:

**Version 1.1.0** (Q2 2024)
* Improved 3D processing algorithms
* Support for multi-channel prediction
* Performance optimizations
* Additional pre-trained models

**Version 1.2.0** (Q3 2024)
* Integration with popular image analysis platforms
* Advanced visualization features
* Distributed processing support
* Model uncertainty estimation

**Version 2.0.0** (2025)
* Next-generation architecture
* Transformer-based models
* Self-supervised pre-training
* Real-time video processing

Versioning Policy
-----------------

U-FISH follows Semantic Versioning:

* **MAJOR** version for incompatible API changes
* **MINOR** version for backwards-compatible new features
* **PATCH** version for backwards-compatible bug fixes

Release Schedule
----------------

* **Patch releases**: As needed for bug fixes
* **Minor releases**: Quarterly
* **Major releases**: Annually

Support Policy
--------------

* Latest version: Full support
* Previous minor version: Security and critical bug fixes
* Older versions: Community support only

For the complete changelog and release notes, visit the `GitHub Releases <https://github.com/GangCaoLab/U-FISH/releases>`_ page. 