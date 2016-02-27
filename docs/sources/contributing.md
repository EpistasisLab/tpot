# Contributing

We welcome you to [check the existing issues](https://github.com/rhiever/tpot/issues/) for bugs or enhancements to work on. If you have an idea for an extension to TPOT, please [file a new issue](https://github.com/rhiever/tpot/issues/new) so we can discuss it.

## How to contribute

The preferred way to contribute to TPOT is to fork the 
[main repository](https://github.com/rhiever/tpot/) on
GitHub:

1. Fork the [project repository](https://github.com/rhiever/tpot):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourLogin/tpot.git
          $ cd tpot

3. Create a branch to hold your changes:

          $ git checkout -b my-contribution

   and start making changes. Never work in the ``master`` branch!

4. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

          $ git add modified_files
          $ git commit

   to record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-contribution

Finally, go to the web page of your fork of the TPOT repo,
and click 'Pull Request' (PR) to send your changes to the maintainers for
review. This will send an email to the maintainers.

(If any of the above seems like magic to you, then look up the 
[Git documentation](http://git-scm.com/documentation) on the web.)

## Before submitting your pull request

Before you submit a pull request for your contribution, please work through this checklist to make sure that you have done everything necessary so we can efficiently review and accept your changes.

If your contribution changes TPOT in any way:

* Update the [documentation](https://github.com/rhiever/tpot/tree/master/docs/sources) so all of your changes are reflected there.

* Update the [README](https://github.com/rhiever/tpot/blob/master/README.md) if anything there has changed.

If your contribution involves any code changes:

* Update the [project unit tests](https://github.com/rhiever/tpot/blob/master/tests.py) to test your code changes.

* Make sure that your code is properly commented with [docstrings](https://www.python.org/dev/peps/pep-0257/) and comments explaining your rationale behind non-obvious coding practices.

* If your code affected any of the pipeline operators, make sure that the corresponding [export functionality](https://github.com/rhiever/tpot/blob/master/tpot/export_utils.py) reflects those changes.

If your contribution requires a new library dependency outside of DEAP and scikit-learn:

* Double-check that the new dependency is easy to install via `pip` or Anaconda and supports both Python 2 and 3. If the dependency requires a complicated installation, then we most likely won't merge your changes because we want to keep TPOT easy to install.

* Add the required version of the library to [.travis.yml](https://github.com/rhiever/tpot/blob/master/.travis.yml#L7)

* Add a line to pip install the library to [.travis_install.sh](https://github.com/rhiever/tpot/blob/master/ci/.travis_install.sh#L46)

* Add a line to print the version of the library to [.travis_install.sh](https://github.com/rhiever/tpot/blob/master/ci/.travis_install.sh#L58)

* Similarly add a line to print the version of the library to [.travis_test.sh](https://github.com/rhiever/tpot/blob/master/ci/.travis_test.sh#L17)

## After submitting your pull request

After submitting your pull request, [Travis-CI](https://travis-ci.com/) will automatically run unit tests on your changes and make sure that your updated code builds and runs on Python 2 and 3. We also use services that automatically check code quality and test coverage.

Check back shortly after submitting your pull request to make sure that your code passes these checks. If any of the checks come back with a red X, then do your best to address the errors.
