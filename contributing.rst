=======================
Contributing to Covasim
=======================

Welcome! We are thrilled you are interested in contributing to Covasim.
This document will help you get started.

Please see the guidelines below to provide feedback or contribute to Covasim
code.  Note that we make no representations that the code works as intended or
that we will provide support, address issues that are found, or accept pull
requests.

Getting started
===============

Contributions to this project are
`released <https://help.github.com/articles/github-terms-of-service/#6-contributions-under-repository-license>`__
to the public under the project's open source license_.

.. _license: https://github.com/InstituteforDiseaseModeling/covasim/blob/master/LICENSE

Note that this project is released with a `Contributor Code of Conduct`_. By participating in this project
you agree to abide by its terms.

.. _Contributor Code of Conduct: https://github.com/InstituteforDiseaseModeling/covasim/blob/master/CODE_OF_CONDUCT.rst

Review the `documentation`_ before you begin. Additionally, this content is also
included in the following README files:

   -  `Project README <https://github.com/InstituteforDiseaseModeling/covasim/blob/master/README.rst>`__: Setup and project overview.
   -  `Covasim README <https://github.com/InstituteforDiseaseModeling/covasim/blob/master/covasim/README.rst>`__: Simulation parameters.
      application.
      docker setup.
   -  `Tests README <https://github.com/InstituteforDiseaseModeling/covasim/blob/master/tests/README.rst>`__: Running tests locally.

Contribution types
===================

This is a fast-moving project with many opportunities to contribute
across the project. We welcome the following types of contributions:

1. Issues:

    * Bug reports.
    * Feature requests.

2. Pull requests:

    * Tests reproducing an issue.
    * Bug fixes.
    * Code to resolve `open approved
      issues <https://github.com/InstituteforDiseaseModeling/covasim/issues?q=is%3Aopen+is%3Aissue+label%3Aapproved>`__.
    * Documentation improvements in the docs_ folder.

.. _docs: https://github.com/InstituteforDiseaseModeling/covasim/tree/master/docs

All external communication about these contribution efforts is currently
occurring on GitHub.


Request new features or report bugs
===================================

If you notice unexpected behavior or a limitation in Covasim, follow the steps below before requesting a new feature or reporting a bug.

1.  First, review the Covasim documentation_ to see if there is already functionality that supports
    what you want to do. In particular, check :doc:`parameters` or  `this
    README <https://github.com/InstituteforDiseaseModeling/covasim/blob/master/covasim/README.md>`_ to see if the behavior you observed might be expected and if configuration options are available to
    provide you with the desired behavior.
2.  Search the existing issues_ to see if there is already one that contains your feedback. If there
    is, **add a thumbs up reaction** to convey your interest in the issue being addressed. This helps the
    team gauge interest without the noise of comments, which trigger notifications to all watchers.
    Comments should be used only if you have new and useful information to share.

.. _documentation: https://institutefordiseasemodeling.github.io/covasim-docs

.. _issues: https://github.com/InstituteforDiseaseModeling/covasim/issues


Open a feature request
----------------------

When opening an issue to request a new feature, do the following:

1.  Provide a clear and descriptive title for the issue.
2.  Include as many details as possible in the body. Fully explain your use case, the problems you're hitting, and the
    solutions you'd like to see to address those problems.

Report a bug
------------

When opening an issue to report a bug, explain the problem and include additional details to help us reproduce the problem:

1.  Describe the specific steps that led to the problem you encountered with as many details as possible.
    Don't just say what you did, but explain how you did it.
2.  Provide specific examples to demonstrate the steps, such as links to files or projects, code snippets,
    or screen shots. Please use Markdown code blocks for code snippets.
3.  Describe the behavior you observed after following the steps and point out exactly what the problem
    with that behavior is, including explaining what you expected to see instead and why.


Submit a pull request
=====================

To contribute directly to Covasim code, do the following:

1.  Fork and clone the Covasim repository.
2.  Install Covasim on your machine. See the `project README`_.
3.  Create a new branch::

        git checkout -b my-branch-name

4.  Make your code changes, including a descriptive commit message.
5.  Push to your fork and submit a pull request.

Although we make no guarantees that a submitted pull request will be accepted, PRs
that meet the following criteria are more likely to be merged:

*   Up-to-date with master with no merge conflicts
*   Self-contained
*   Fix a demonstrable limitation of bug
*   Follow the current code style
*   If the PR introduces a new feature, it has complete `Google style docstrings`_ and comments,
    and a test demonstrating its functionality
*   Otherwise, sample code demonstrating old and new behavior (this can be in the PR comment on
    GitHub, not necessarily committed in the repo)

.. _Google style docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html
.. _project README: https://github.com/InstituteforDiseaseModeling/covasim/blob/master/README.rst

If you have additional questions or comments, contact covasim@idmod.org.

Resources
=========

-  `LitCovid <https://www.ncbi.nlm.nih.gov/research/coronavirus/>`__
-  `MIDAS network <https://midasnetwork.us/covid-19/>`__
-  `How to contribute to open
   source <https://opensource.guide/how-to-contribute/>`__
-  `Using pull
   requests <https://help.github.com/articles/about-pull-requests/>`__
-  `GitHub help <https://help.github.com>`__

