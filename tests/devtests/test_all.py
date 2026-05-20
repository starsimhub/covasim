#!/usr/bin/env pytest
# Run all dev tests. Note: these are not guaranteed to work.

import sciris as sc
import runpy
import pytest
import pylab as pl

pl.switch_backend('agg') # To avoid graphs from appearing -- if you want them, run the scripts directly
scripts = sc.getfilelist(ext='py')

script_str= "\n".join(scripts)
print(f'Running tests on:\n{script_str}...')

@pytest.mark.parametrize('script', scripts)
def test_script_execution(script):
    runpy.run_path(script)