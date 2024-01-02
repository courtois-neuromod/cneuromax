.. _contribution:

************
Contribution
************

Link to the GitHub repository: https://github.com/courtois-neuromod/cneuromax.

.. note::

    Make sure to have completed the installation steps before moving to
    this section.

There are many ways to interact with the code base, with ranging levels of
collaborativity. The following instructions are meant for people who wish to
contribute to the code base, either by fixing bugs, adding new features or
improving the documentation.

Regardless of whether you wish to contribute to the ``main`` branch or solely
to your own branch, check out the ``classify_mnist`` `code folder
<https://github.com/courtois-neuromod/cneuromax/tree/main/cneuromax/task/classify_mnist>`_
for a template of how to structure your code.

Making sure the code doesn't break
----------------------------------

The ``main`` branch is protected meaning that contributions happen through
pull requests rather than direct pushes.

In order for any pull request to go through, it will need to pass a number of
common and standard checks (using GitHub Actions) that ensure that the code is
of high quality and does not break any portion of the existing code base.

.. note::

    Do not be intimidated by the number of checks, we will later introduce how to
    seemlessly integrate them in your workflow.

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

* Making sure that the Docker image can still be built.

* Making sure that the documentation can still be built and pushed to Github
  Pages.

Testing locally
---------------

Rather than having to wait for the GitHub tests to finish verifying your code,
you can run some of the tests locally to debug preventively.

**Fast Tests (sub 1 second)**

Formatting and linting tests are quick to run. The most common way to execute
them is through pre-commit hooks. We use the `pre-commit
<https://pre-commit.com/#quick-start>`_ library, which configure to run
formatting and linting tests upon each ``git commit`` command, preventing the
commit from going through if the tests fail.

.. note::

    To disable this behaviour (for instance when you're commiting to a
    dev/local branch and don't want to deal with formatting / code validity,
    you can instead run ``git commit --no-verify``.

**Slow Tests**

Unit-tests and typecheck tests are slower, hence not suitable to run upon each
commit. However they are pretty useful to save time right before merging your
branch. To run those tests, you can use the following commands:

.. code-block:: bash

    pytest cneuromax
    mypy --config-file=pyproject.toml cneuromax


Setting up VSCode
-----------------

Rather than being welcomed to a red wave of errors and warnings every time you
``git commit`` or run the slow tests, we suggest that you set up your editor to
notify you of most issues before you commit.

We provide a ``.devcontainer.json`` file that allows you to develop locally
using VSCode and the repository's Docker image (it should not be too hard to
adapt this file to other IDEs). In order to use it, you will need to install
the `Remote - Containers` extension.

.. code-block:: bash

    code --install-extension ms-vscode-remote.remote-containers

From then on, upon opening the repository in VSCode, you should be prompted to
open the repository in a container. If you are not, you can open the command
palette (``Ctrl+Shift+P``) and search for
``Remote-Containers: Reopen in Container``.

There are so far two small pain points:

- The very first time you boot-up the Dev Container, some extensions won't be
  loaded correctly. A window could pop-up asking you to reload the window which
  you will need to do. If it doesn't pop up, open the command palette
  (``Ctrl+Shift+P``) and search for ``Developer: Reopen folder locally`` before
  reopening the Dev Container.

- The esbonio server will sometimes announce a build error (bottom right),
  which will prevent further documentation visualization. To fix this, you
  should delete the contents of the ``docs/_build`` and ``docs/_autosummary``
  folders (do not delete the folders themselves if you use Dropbox/Maestral)
  and restart the esbonio server (by its icon).

GitHub Copilot is installed in the DevContainer. Simply discard the log-in
notifications if you do not want to make use of it.
You can run ``git``, ``pytest`` & ``mypy`` commands from the integrated
terminal. However running the library itself requires special Docker flags and
should thus be ran from the terminal outside of VSCode (refere to the
``Execution`` section).

Git/GitHub workflow for contributing
------------------------------------

In a terminal window, change directory to the cneuromax repository.

.. code-block:: bash

    cd ${CNEUROMAX_PATH}

.. note::

    You can avoid typing the following commands by hand by using the VSCode
    ``Source Control`` tab on the left and the branch tab on the bottom left.
    To commit with no verification, press the ``...`` button above the
    ``Commit`` button and select ``Commit All (No Verify)``.

Create a new branch for your contribution.

.. code-block:: bash

    git checkout main
    git pull
    git checkout -b <YOUR_BRANCH_NAME>

Make your changes, commit them and push them to the remote repository.

.. code-block:: bash

    git add .
    git commit -m "<COMMIT_MESSAGE>" # can add the --no-verify flag to skip tests
    git push

If you are done with your contribution, you can create a pull request on
GitHub. If new changes have been introducted to the ``main`` branch while you
were working on your development branch, you will need to update your branch
with the latest changes from ``main``, you can do so by running the following
commands.

.. code-block:: bash

    git checkout main
    git pull
    git checkout <YOUR_BRANCH_NAME>
    git merge main # might need to resolve conflicts (easier to do in VSCode)
    git push

Once you have pushed your changes, you can create a pull request on GitHub.
Once it is approved, delete your branch and make sure to pull the changes to
your local repository.

.. code-block:: bash

    git checkout main
    git pull
    git branch -d <YOUR_BRANCH_NAME>

Documenting your contribution
-----------------------------

.. note::

    Make sure to not leave any of your ``__init__.py`` files empty else the
    specific subpackage will not be documented.

We use `sphinx.ext.autosummary
<https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_ to
automatically generate documentation from `Google-style Python docstrings
<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
This webpage holds the API reference documentation for the ``main`` branch of
the repository and is automatically updated upon each push.
Take a look at `this Python file
<https://github.com/courtois-neuromod/cneuromax/blob/main/cneuromax/fitting/deeplearning/datamodule/base.py>`_
and its `corresponding documentation webpage
<https://courtois-neuromod.github.io/cneuromax/cneuromax.fitting.deeplearning.datamodule.base.html>`_
that showcase most of the available docstring commands available and their
effects on the documentation page.

.. note::

    Document your ``__init__`` method arguments in the class docstring rather
    than in the ``__init__`` docstring.

Assuming that you are using the library's development Docker image in your
editor, you can preview your changes to ``.rst`` by clicking the preview button
on the top right of the editor. In general, you can preview your changes to all
``.rst``, ``.py`` and ``README.md`` files after re-building the documentation
by pressing the ``esbonio`` button on the bottom right of the editor and then
opening the locally created ``.html`` files.

Setting up Maestral/Dropbox to move code across machines
-----------------------------------------------------------

Rather than having to manually move code across machines, we suggest that you
use a Dropbox folder to automatically sync your code across machines.

On machines where you have root access, you can simply install Dropbox.
On machines where you do not have root access, you can install Maestral as a
drop-in replacement for Dropbox (Make sure not to install both Dropbox and
Maestral on the same machine).

.. code-block:: bash

    tmux
    module load python/3.10
    pip install -U maestral
    python -m maestral start

You will be prompted the following question: **How would you like to you link
your account?**

Choose: **Print auth URL to console**

Open the URL and press Allow.

Copy the code that appears in the browser.

**Enter the auth code:** Paste.

**Please choose a local Dropbox folder:**  ``/scratch/<USER>/Dropbox``

Would you like sync all folders? **No**

Choose which folders to include: **cneuromax**

You can now close the console window (``Ctrl+B``, ``D``) and the
synchronization will continue in the background.

You can reattach to the console window and check the status of the
synchronization by running:

.. code-block:: bash

    tmux attach -t 0 # The number is the index of the window
    python -m maestral status

Finally, there are some files that you probably do not want to sync across
all machines. On a machine with Dropbox, run:

.. code-block:: bash

    mkdir -p data/ docs/_build/ docs/_autosummary/ .vscode/ .coverage
    mkdir -p .mypy_cache/ .pytest_cache/ .ruff_cache/
    sudo attr -s com.dropbox.ignored -V 1 data/
    sudo attr -s com.dropbox.ignored -V 1 docs/_build/
    sudo attr -s com.dropbox.ignored -V 1 docs/_autosummary/
    sudo attr -s com.dropbox.ignored -V 1 .vscode/
    sudo attr -s com.dropbox.ignored -V 1 .coverage
    sudo attr -s com.dropbox.ignored -V 1 .mypy_cache/
    sudo attr -s com.dropbox.ignored -V 1 .pytest_cache/
    sudo attr -s com.dropbox.ignored -V 1 .ruff_cache/

On a machine with Maestral, edit your `.mignore` file to exclude the files you
do not want to sync.

Example of the contents of a `.mignore` file:

.. code-block:: python

    **/data

Freezing the repositories for publication
-----------------------------------------

For your code to remain reproducible after publication, we suggest that you
create a new branch or fork the repository.

If you want to freeze and make your branch/fork of this repository as light as
possible, you can delete the following:

- Any non-relevant folder inside ``cneuromax/fitting/deeplearning/datamodule/``
- Any non-relevant folder inside ``cneuromax/fitting/deeplearning/litmodule/``
- Any non-relevant folder inside ``cneuromax/fitting/deeplearning/nnmodule/``
- If you are not doing Neuroevolution, the
  ``cneuromax/fitting/neuroevolution/`` folder
- The ``cneuromax/serving/`` folder
- Any non-relevant folder inside ``cneuromax/task/``
- The ``docs/`` folder
- The ``Dockerfile`` file
- Most of the contents of the ``README.md`` file
- The ``renovate.json`` file
- The irrelevant dependencies in the ``pyproject.toml`` file
