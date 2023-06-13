.. _contribution:

************
Contribution
************

.. note::

    Make sure to have completed the installation steps before moving to
    this section.

Link to the GitHub repository: https://github.com/courtois-neuromod/cneuroml.

Developping locally
-------------------

In order to develop locally, you will need to install the development
dependencies. This can be done by running the following commands:

.. code-block:: console

    $ cd ${CNEUROML_PATH}
    $ python -m venv venv
    $ . venv/bin/activate
    $ # Install OpenMPI if it's not already installed
    $ sudo apt install -y libopenmpi-dev
    $ pip install --extra-index-url https://download.pytorch.org/whl/cpu -e .[dev]

You now have all the dependencies needed to develop locally (do note that
these libraries and their version are not meant for actually running the
code, to do so it is preferable to work with containers for reproducibility
purposes).

Making sure the code doesn't break
----------------------------------

The main branch is protected meaning that contributions happen through
pull requests rather than direct pushes.

In order for any pull request to go through, it will need to pass a number of
common and standard checks that ensure that the code is of high quality and
does not break any portion of the existing code base.

Those checks are:

* Making sure that the Python code follows a clean common format and is
  PEP8 compliant. In order to do so we make use of the following libraries:

  * Formatting: `black
    <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_
    (79 lines for code, 72 for comments)
  * Linting: `ruff <https://beta.ruff.rs/docs/tutorial/#getting-started>`_
    (all rules that don't create conflicts with existing libraries)
  * Type checking: `mypy
    <https://mypy.readthedocs.io/en/stable/getting_started.html>`_ (strict)

* Making sure that the Python code passes all of the existing unit-tests. If
  they do not, it means that your code is breaking a portion of the
  code base. We currently only make use of unit-tests (leveraging `pytest
  <https://docs.pytest.org/en/7.3.x/getting-started.html>`_) but will most
  likely eventually develop tests with broader purposes.

* Making sure that the reST documentation files are formatted and linted
  (using `doc8 <https://github.com/PyCQA/doc8>`_ 79 lines).

* Making sure that the YAML files are formatted and linted
  (using `yamllint
  <https://yamllint.readthedocs.io/en/stable/quickstart.html#running-yamllint>`_).

* Making sure that there are no trailing whitespaces and that all files,
  regardless of the extension end with a newline.

* If any change is made to the ``pyreqs/`` or ``containers/`` folder, that the
  Docker/Podman image can still be built.

* If any change is made to the ``docs/`` or ``cneuroml/`` folder, that the
  documentation can still be built and pushed to Github Pages.

Testing locally
---------------

Rather than having to wait for the GitHub CI (continuous integration) tests to
finish verifying your code, you can run some of the tests locally.

**Fast Tests (sub 1 second)**

Formatting and linting tests are quick to run. The most common way to execute
them is through pre-commit hooks. We use the
`pre-commit <https://pre-commit.com/#quick-start>`_ library, which after being
installed runs formatting and linting tests upon each ``git commit`` command,
preventing the commit from going through if the tests fail.

.. code-block:: console

    $ cd ${CNEUROML_PATH}
    $ . venv/bin/activate
    $ pre-commit install

From now on, ``git commit`` commands in this repository will automatically make
sure that all important files are functional and well formatted.

.. note::

    To disable this behaviour (for instance when you're commiting to a
    dev/local branch and don't want to deal with formatting / code validity,
    you can instead run ``git commit --no-verify``.

**Slow Tests**

Unit-tests and typecheck tests are slower, hence not suitable to run upon each
commit. However they are pretty useful to save time right before merging your
branch. To run those tests, you can use the following commands:

.. code-block:: console

    $ cd ${CNEUROML_PATH}
    $ . venv/bin/activate
    $ pytest cneuroml
    $ mypy --strict cneuroml


Setting up VSCode
-----------------

Rather than being welcomed to a red wave of errors and warnings every time you
``git commit`` or run the slow tests, we suggest that you set up your editor to
notify you of any issues before you commit.

On VSCode, this means installing the following extensions:

* `ms-python.python
  <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_
* `ms-python.black-formatter
  <https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter>`_
* `charliermarsh.ruff
  <https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_
* `lextudio.restructuredtext
  <https://marketplace.visualstudio.com/items?itemName=lextudio.restructuredtext>`_
  (``doc8`` enabler)
* `fnando.linter
  <https://marketplace.visualstudio.com/items?itemName=fnando.linter>`_
  (``yamllint`` enabler)

And here are the settings to insert into your
``~/.config/Code/User/settings.json`` file.

.. code-block:: json

    // Ruff
    "ruff.args": [
        "--config=pyproject.toml"
    ],
    // Black
    "black-formatter.args": [
        "--config=pyproject.toml"
    ],
    // Black + Ruff
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    },
    // MyPy
    "python.linting.mypyEnabled": true,
    "python.linting.mypyArgs": [
        "--config-file=pyproject.toml"
    ],
    // ReStructuredText
    "esbonio.sphinx.buildDir" : "${workspaceFolder}/docs/_build/html",
    "esbonio.sphinx.confDir"  : "${workspaceFolder}/docs",
    "esbonio.sphinx.srcDir"   : "${workspaceFolder}/docs",
    "restructuredtext.linter.doc8.executablePath": "/home/max/.local/bin/doc8",
    "restructuredtext.syntaxHighlighting.disabled": true,
    // Miscellaneous
    "files.insertFinalNewline": true,
    "files.trimTrailingWhitespace": true,
    // Pylance
    "python.analysis.ignore": [
        "/**"
    ],

.. note::

    Pylance throws some incorrect errors with pytest and is otherwise
    redundant due to using mypy, so we suggest disabling its analysis
    (even though the setting is called ``python.analysis.ignore``, it actually
    only applies to Pylance). We suggest not uninstalling it, as it is
    has other useful features like syntax highlighting.

Git/GitHub workflow for contributing
------------------------------------

In a terminal window, change directory to the cneuroml repository.

.. code-block:: console

    $ cd ${CNEUROML_PATH}

Create a new branch for your contribution.

.. code-block:: console

    $ git checkout main
    $ git pull
    $ git checkout -b <YOUR_BRANCH_NAME>

Make your changes, commit them and push them to the remote repository.

.. code-block:: console

    $ git add .
    $ git commit -m "<COMMIT_MESSAGE>"
    $ git push

If you are done with your contribution, you can create a pull request on
GitHub. If new changes have been introducted to the ``main`` branch while you
were working on your development branch, you will need to update your branch
with the latest changes from ``main``, you can do so by running the following
commands.

.. code-block:: console

    $ git checkout main
    $ git pull
    $ git checkout <YOUR_BRANCH_NAME>
    $ git rebase main

Now, create a pull request on GitHub, once it is approved, delete your branch
and pull the changes to your local repository.

.. code-block:: console

    $ git checkout main
    $ git pull origin main
    $ git branch -d <YOUR_BRANCH_NAME>

Freezing the repositories for publication
-----------------------------------------

For your code to remain reproducible after publication, we suggest that you
create a new branch or fork the repository.

TODO: Add pruning instructions.
