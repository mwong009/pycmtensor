# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.
You can contribute in many ways:

## Types of Contributions

### Bug report

Report bugs at https://github.com/mwong009/pycmtensor/issues.

If you find a bug or other unexpected behavior while using PyCMTensor, open an issue on the GitHub repository and we will try to respond and (hopefully) solve the problem in a timely manner.
Similarly, if you have a feature request or question about the library, the best place to post those is currently on GitHub as an issue.
If you are reporting a bug, please include:

- Your operating system name and Python version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Bugfixes

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

PyCMTensor could always use more documentation, whether as part of the
official PyCMTensor docs, in docstrings, etc.

## Get Started

Ready to contribute? Here's how to set up `pycmtensor` as a developer.

1. Fork the `pycmtensor` repo on GitHub.

2. Clone your fork locally:

   ```
   git clone git@github.com:your_name_here/pycmtensor.git
   ```

3. Install your local forked copy into a virtualenv. We suggest using [Poetry](https://python-poetry.org/) to manage the dependencies:

   ```
   cd pycmtensor
   conda env create -f environment/environment_<[windows|macos|linux]>.yml
   conda activate pycmtensor-dev
   pip install poetry
   poetry install -E dev
   ```

4. Create a branch for local development:

   ```
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass `black`, `isort` and the
   tests.
   
   ```
   poetry run black .
   poetry run isort .
   poetry run pytest
   ```

7. Commit your changes and push your branch to GitHub:

   ```
   cz c -s
   <select the options and write your commit message>
   git push
   ```

   Here we use commitizen to mange the Git commit messages. Make sure that the format of the commit message follows the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should pass all tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add it to the Pull Request description.
3. The pull request should work for Python 3.9 or higher, and make sure that the tests pass for all supported Python versions.
