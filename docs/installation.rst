Installation
============

Installing with uv (Recommended)
---------------------------------

We recommend using ``uv`` for package management. 

First, install ``uv`` if you haven't already:

.. code-block:: bash

   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

Then install ``frizzle``:

.. code-block:: bash

   uv add frizzle

This will install ``frizzle`` and all its dependencies in your project.

Alternative Installation Methods
---------------------------------

Using pip
~~~~~~~~~

If you prefer to use `pip`, we won't judge too harshly:

.. code-block:: bash

   pip install frizzle

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

To install the latest development version from source:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/andycasey/frizzle.git
   cd frizzle

   # Install with uv
   uv add .

   # Or with pip
   pip install .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development, you can install `frizzle` in editable mode with development dependencies:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/andycasey/frizzle.git
   cd frizzle

   # Install in development mode
   uv add --editable .

   # Or with pip
   pip install -e ".[dev]"

Verifying Installation
----------------------

To verify that frizzle is installed correctly, you can run:

.. code-block:: python

   import frizzle
   print(frizzle.__version__)

This should print the version number without any errors.
