"""
Pytest configuration for the Covasim test suite.

``devtests/`` holds developer scratch tests; ``collect_ignore`` stops pytest from
recursing into it when a bare ``pytest`` / ``pytest .`` is run.
"""
collect_ignore = ['devtests']
