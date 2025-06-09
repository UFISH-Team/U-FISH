# U-FISH Documentation

This directory contains the source files for U-FISH's documentation, built with Sphinx and hosted on Read the Docs.

## Building Documentation Locally

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Building HTML Documentation

On Unix/macOS:
```bash
make html
```

On Windows:
```bash
make.bat html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

### Other Build Formats

```bash
# Build PDF (requires LaTeX)
make latexpdf

# Build EPUB
make epub

# Clean build directory
make clean
```

## Documentation Structure

- `index.rst` - Main documentation entry point
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start tutorial
- `user_guide.rst` - Comprehensive user guide
- `api_reference.rst` - API documentation
- `cli_reference.rst` - CLI documentation
- `tutorials.rst` - Step-by-step tutorials
- `benchmarks.rst` - Performance benchmarks
- `contributing.rst` - Contribution guidelines
- `changelog.rst` - Version history
- `conf.py` - Sphinx configuration
- `requirements.txt` - Documentation dependencies

## Writing Documentation

### ReStructuredText

The documentation is written in reStructuredText (RST). See the [Sphinx RST Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for syntax reference.

### API Documentation

API documentation is automatically generated from docstrings using autodoc. Follow NumPy style docstrings:

```python
def function(param1, param2):
    """
    Short description.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
        
    Returns
    -------
    type
        Description of return value.
    """
```

### Adding New Pages

1. Create a new `.rst` file
2. Add it to the `toctree` in `index.rst`
3. Build and verify the documentation

## Read the Docs

The documentation is automatically built and hosted on Read the Docs when changes are pushed to the main branch.

Configuration is in `.readthedocs.yaml` at the repository root.

## Live Development

For live reloading during documentation development:

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

Then open http://127.0.0.1:8000 in your browser. 