# Contributing

We welcome you to check the existing issues for bugs or enhancements to work on. If you have an idea for an extension to TPOT, please file a new issue so we can discuss it.

# Contribution Guide

We welcome you to [check the existing issues](https://github.com/EpistasisLab/tpot/issues/) for bugs or enhancements to work on. If you have an idea for an extension to TPOT, please [file a new issue](https://github.com/EpistasisLab/tpot/issues/new) so we can discuss it.

## Project layout

Both the latest stable release and the development version of TPOT are on the [main branch](https://github.com/EpistasisLab/tpot/tree/main). Ensure you are working from the correct commit when contributing.

In terms of directory structure:

* All of TPOT's code sources are in the `tpot` directory
* The documentation sources are in the `docs_sources` directory
* Images in the documentation are in the `images` directory
* Tutorials for TPOT are in the `tutorials` directory
* Unit tests for TPOT are in the `tests.py` file

Make sure to familiarize yourself with the project layout before making any major contributions, and especially make sure to send all code changes to the `main` branch.

## How to contribute

The preferred way to contribute to TPOT is to fork the
[main repository](https://github.com/EpistasisLab/tpot/) on
GitHub:

1. Fork the [project repository](https://github.com/EpistasisLab/tpot):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourUsername/tpot.git
          $ cd tpot

3. Create a branch to hold your changes:

          $ git checkout main
          $ git pull origin main
          $ git checkout -b my-contribution

4. Make sure your local environment is setup correctly for development. Set up your local environment following the [Developer Installation instructions](installing.md#developerlatest-branch-installation). Ensure you also install `pytest` (`conda install pytest` or `pip install pytest`) into your development environment so that you can test changes locally.

5. Start making changes on your newly created branch. Work on this copy on your computer using Git to do the version control.

6. Check your changes haven't broken any existing tests and pass all your new tests. Navigate the terminal into the `tpot/tpot/` folder and run the command `pytest` to start all tests. (note, you must have the `pytest` package installed within your dev environment for this to work):

          $ pytest

7. When you're done editing and local testing, run:

          $ git add modified_files
          $ git commit

   to record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-contribution

Finally, go to the web page of your fork of the TPOT repo, and click 'Pull Request' (PR) to send your changes to the maintainers for review. Make sure that you send your PR to the `main` branch. This will start the CI server to check all the project's unit tests run and send an email to the maintainers.

(If any of the above seems like magic to you, then look up the
[Git documentation](http://git-scm.com/documentation) on the web.)

## Before submitting your pull request

Before you submit a pull request for your contribution, please work through this checklist to make sure that you have done everything necessary so we can efficiently review and accept your changes.

If your contribution changes TPOT in any way:

* Update the [documentation](https://github.com/EpistasisLab/tpot/tree/main/docs) so all of your changes are reflected there.

* Update the [README](https://github.com/EpistasisLab/tpot/blob/main/README.md) if anything there has changed.

If your contribution involves any code changes:

* Update the [project unit tests](https://github.com/EpistasisLab/tpot/tree/main/tpot/tests) to test your code changes.

* Make sure that your code is properly commented with [docstrings](https://www.python.org/dev/peps/pep-0257/) and comments explaining your rationale behind non-obvious coding practices.


If your contribution requires a new library dependency:

* Double-check that the new dependency is easy to install via `pip` or Anaconda. If the dependency requires a complicated installation, then we most likely won't merge your changes because we want to keep TPOT easy to install.


## After submitting your pull request

After submitting your pull request, GitHub will automatically run unit tests on your changes and make sure that your updated code builds and runs. We also use services that automatically check code quality and test coverage.

Check back shortly after submitting your pull request to make sure that your code passes these checks. If any of the checks come back with a red X, then do your best to address the errors.
