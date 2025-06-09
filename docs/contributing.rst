Contributing Guide
==================

We welcome contributions to U-FISH! This guide will help you get started.

How to Contribute
-----------------

There are many ways to contribute to U-FISH:

* **Report bugs** and request features via GitHub Issues
* **Improve documentation** by submitting pull requests
* **Submit bug fixes** or implement new features
* **Add new datasets** or improve existing ones
* **Optimize performance** or add new algorithms
* **Create tutorials** or example notebooks

Development Setup
-----------------

Setting Up Your Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/U-FISH.git
      cd U-FISH

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Project Structure
~~~~~~~~~~~~~~~~~

.. code-block:: text

   U-FISH/
   ├── ufish/              # Main package
   │   ├── api.py         # High-level API
   │   ├── cli.py         # Command-line interface
   │   ├── data.py        # Data loading utilities
   │   ├── model/         # Model architectures
   │   ├── unet/          # U-Net implementation
   │   └── utils/         # Utility functions
   ├── tests/             # Test suite
   ├── notebooks/         # Example notebooks
   ├── benchmark/         # Benchmarking scripts
   └── docs/              # Documentation

Code Style
----------

We follow PEP 8 and use several tools to maintain code quality:

* **Black** for code formatting
* **isort** for import sorting
* **flake8** for linting
* **mypy** for type checking

Run all checks:

.. code-block:: bash

   # Format code
   black ufish tests
   
   # Sort imports
   isort ufish tests
   
   # Run linter
   flake8 ufish tests
   
   # Type checking
   mypy ufish

Testing
-------

Writing Tests
~~~~~~~~~~~~~

Add tests for new features in the ``tests/`` directory:

.. code-block:: python

   # tests/test_new_feature.py
   import pytest
   from ufish import new_feature
   
   def test_new_feature():
       """Test the new feature"""
       result = new_feature(input_data)
       assert result == expected_output
   
   @pytest.mark.parametrize("input,expected", [
       (input1, expected1),
       (input2, expected2),
   ])
   def test_new_feature_parametrized(input, expected):
       """Test with multiple inputs"""
       assert new_feature(input) == expected

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/test_api.py
   
   # Run with coverage
   pytest --cov=ufish --cov-report=html
   
   # Run only fast tests
   pytest -m "not slow"

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Documentation is built using Sphinx:

.. code-block:: bash

   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

* Use reStructuredText format
* Include code examples
* Add docstrings to all public functions:

  .. code-block:: python

     def predict(self, image, threshold=0.5):
         """
         Predict FISH spots in an image.
         
         Parameters
         ----------
         image : np.ndarray
             Input image (2D or 3D).
         threshold : float, optional
             Detection threshold between 0 and 1.
             
         Returns
         -------
         spots : pd.DataFrame
             Detected spots with columns [y, x] or [z, y, x].
         enhanced : np.ndarray
             Enhanced image.
             
         Examples
         --------
         >>> spots, enhanced = ufish.predict(image)
         >>> print(f"Found {len(spots)} spots")
         """

Pull Request Process
--------------------

1. **Create a branch** for your feature:

   .. code-block:: bash

      git checkout -b feature/amazing-feature

2. **Make your changes** and commit:

   .. code-block:: bash

      git add .
      git commit -m "Add amazing feature"

3. **Write/update tests** for your changes

4. **Update documentation** if needed

5. **Run all checks**:

   .. code-block:: bash

      # Format and lint
      black ufish tests
      isort ufish tests
      flake8 ufish tests
      
      # Run tests
      pytest
      
      # Build docs
      cd docs && make html

6. **Push your branch**:

   .. code-block:: bash

      git push origin feature/amazing-feature

7. **Create a Pull Request** on GitHub

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

* **Title**: Use a clear, descriptive title
* **Description**: Explain what changes you made and why
* **Tests**: Ensure all tests pass
* **Documentation**: Update relevant documentation
* **Changelog**: Add an entry to CHANGELOG.md

Adding New Features
-------------------

Model Architecture
~~~~~~~~~~~~~~~~~~

To add a new model architecture:

1. Create a new file in ``ufish/model/``:

   .. code-block:: python

      # ufish/model/new_model.py
      import torch
      import torch.nn as nn
      
      class NewModel(nn.Module):
          def __init__(self, in_channels=1, out_channels=1):
              super().__init__()
              # Define layers
              
          def forward(self, x):
              # Implement forward pass
              return output

2. Register in ``ufish/api.py``:

   .. code-block:: python

      from .model.new_model import NewModel
      
      MODEL_REGISTRY = {
          'ufish_2d': UNet2D,
          'new_model': NewModel,
      }

Data Formats
~~~~~~~~~~~~

To support new data formats:

1. Add loader in ``ufish/utils/io.py``:

   .. code-block:: python

      def read_new_format(filepath):
          """Read data in new format"""
          # Implementation
          return data

2. Update ``ufish/data.py`` to handle the format

CLI Commands
~~~~~~~~~~~~

To add new CLI commands:

1. Add command in ``ufish/cli.py``:

   .. code-block:: python

      @cli.command()
      @click.argument('input_file')
      @click.option('--param', default=1.0)
      def new_command(input_file, param):
          """Description of new command"""
          # Implementation

Reporting Issues
----------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

* U-FISH version (``ufish.__version__``)
* Python version
* Operating system
* Minimal code example reproducing the issue
* Full error traceback
* Sample data if possible

Feature Requests
~~~~~~~~~~~~~~~~

For feature requests, please describe:

* The problem you're trying to solve
* Your proposed solution
* Alternative solutions you've considered
* Any relevant examples or references

Community Guidelines
--------------------

* Be respectful and inclusive
* Provide constructive feedback
* Help others when you can
* Follow the code of conduct

Getting Help
------------

* **Documentation**: https://u-fish.readthedocs.io
* **GitHub Issues**: https://github.com/GangCaoLab/U-FISH/issues
* **Discussions**: https://github.com/GangCaoLab/U-FISH/discussions

Recognition
-----------

Contributors will be:

* Listed in the AUTHORS file
* Mentioned in release notes
* Acknowledged in publications using their contributions

Thank you for contributing to U-FISH! 