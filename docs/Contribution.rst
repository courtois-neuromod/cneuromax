.. _contribution:

************
Contribution
************

Link to the GitHub repository: https://github.com/courtois-neuromod/cneuroml.

The main branch is protected meaning that contributions happen through pull
requests rather than direct pushes.

Making sure the code doesn't break
----------------------------------

In order for a pull request to be accepted, it will need to pass a number
of common and standard checks that ensure that the code is of high quality and
does not break the existing code base.

Those checks are:

* Making sure that your Python code is properly linted & PEP8 compliant. To do
  so, we make use of **black** (79 lines for code, 72 for comments) and **ruff**.

* Making sure that your Python code passes all of the unit tests. If
  they do not, it means that your code is breaking a portion of the existing code
  base.

* Making sure that there are no trailing whitespaces and that all files,
  regardless of the extension end with a newline.

* If any change is made to the **pipreqs** or **docker** folder, that the
  Docker/Podman image can still be built.

* If any change is made to the **docs** or **cneuroml** folder, that the
documentation can still be built and pushed to ReadTheDocs.

In order to facilitate the contribution process, we therefore suggest that you
install & setup **black**, **ruff** and **pre-commit**.

.. code-block:: console

    $ cd ${CNEUROML_PATH}
    $ pip install black[jupyter] ruff pre-commit
    $ pre-commit install

From now on, **git commit** will automatically make sure that your code is
properly linted and that there are no trailing whitespaces or missing newlines.
To disable this behaviour, you can instead run **git commit --no-verify**.

(Optional) Setting up VSCode
----------------------------

To use Ruff in VSCode, install the `official extension
<https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`_.

In order to make sure that your files are properly formatted on save, make sure
the following is present in your **~/.config/Code/User/settings.json** file.???

.. code-block:: json

    "[python]": {
        "editor.formatOnType": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        },
    },
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--config=pyproject.toml"
    ],
    "editor.formatOnSave": true,
    "ruff.args": [
        "--config=pyproject.toml"
    ],
    "files.insertFinalNewline": true,
    "files.trimTrailingWhitespace": true,

Git/GitHub workflow for contributing
------------------------------------

In a terminal window, change directory to the cneuroml repository.

.. code-block:: console

    $ cd ${CNEUROML_PATH}

Create a new branch for your contribution.

.. code-block:: console

    $ git pull origin main
    $ git checkout -b <BRANCH_NAME>

Make your changes, commit them and push them to the remote repository.

.. code-block:: console

    $ git add .
    $ git commit -m "<COMMIT_MESSAGE>"
    $ git push origin <BRANCH_NAME>

Now, create a pull request on GitHub, once it is approved, delete your branch
and pull the changes to your local repository.

.. code-block:: console

    $ git checkout main
    $ git pull origin main
    $ git branch -d <BRANCH_NAME>

Freezing the repositories for publication
-----------------------------------------

For your code to remain reproducible after publication, we suggest that you
create a new branch or fork the repository.
