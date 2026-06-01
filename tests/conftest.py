"""
Pytest configuration for the Covasim test suite.

During the v4.0 Starsim port, v3 tests that exercise removed/replaced APIs are
quarantined under ``tests/_legacy/`` (including the former ``unittests/`` suite at
``tests/_legacy/unittests/``). ``collect_ignore`` stops pytest from recursing into
those directories when a bare ``pytest`` / ``pytest .`` is run, so the quarantine
neither runs nor errors. ``devtests/`` holds developer scratch tests and is
likewise excluded. The quarantines are restored over M2-M10 and deleted at M10.
"""
collect_ignore = ['_legacy', 'devtests']
