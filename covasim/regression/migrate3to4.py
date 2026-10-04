'''
Migrate scripts written for Covasim v3 to Covasim v4.

Most v3 code runs unchanged in v4. This script applies the few changes that are
mechanical (e.g. ``sim.t`` → ``sim.ti``), and lists the lines that may need a
change that requires judgment (e.g. code that assumes agents who have died stay
in the per-agent arrays). The rules are described in ``docs/migrate3to4.md``.

By default, the changes are shown but not made; use ``--apply`` to change the files.

**Examples**::

    covasim-migrate3to4 my_script.py           # Show the changes for one file
    covasim-migrate3to4 my_folder --apply      # Change every .py file and notebook in a folder
'''

import re
import json
import difflib
import argparse
import sciris as sc

__all__ = ['rules', 'checks', 'migrate_findinds', 'migrate_code', 'migrate_file', 'migrate', 'main']


# Any variable that is a sim, e.g. "sim", "self.sim", "base_sim", "msim.base_sim"
sim = r'\b(\w*sim)'

# The mechanical rules: each is a regular expression, its replacement, and a description
rules = [
    [sim + r'\.t\b(?![\w(.]| *=[^=])',                           r'\1.ti',                       'sim.t → sim.ti (sim.t is now the Starsim timeline)'],
    [sim + r'\.pars\[',                                          r'\1[',                         "sim.pars['key'] → sim['key'] (the COVID parameters are no longer in sim.pars)"],
    [sim + r'''\.results\[['"](transtree|agehist|gen_time)['"]\]''', r'\1.\2',                   "sim.results['transtree'] → sim.transtree (results can only hold time series)"],
    [sim + r'\.results\.(transtree|agehist|gen_time)\b',         r'\1.\2',                       'sim.results.transtree → sim.transtree (results can only hold time series)'],
]
findinds_rule = 'sc.findinds(people.x) → cv.true(people.x) (sc.findinds() returns positions rather than UIDs, which differ once agents have died)'

# The checks: each is a regular expression and the reason the line may need to be changed by hand
checks = [
    [r'len\((\w+\.)*(people|ppl)\)|' + sim + r'\.n\b',           'Only counts agents who are alive; use sim.people.n_uids for the number ever created, and sim.people.indices() rather than np.arange(len(sim.people))'],
    [r'(people|ppl)\.(dead|date_dead)\b',                        'people.dead and people.date_dead include agents who have died, but other arrays (e.g. people.age) do not; indexing by UID works (people.age[cv.true(people.dead)]), but to combine them elementwise, use .raw (e.g. people.age.raw[:people.n_uids])'],
    [r'np\.(nonzero|flatnonzero|where|argwhere)\([^\n]*(people|ppl)\.', 'Returns positions rather than UIDs, which differ once agents have died; use cv.true() or .uids'],
    [r'''\b(popfile|load_pop|save_pop) *=[^=]|cv\.make_(people|randpop|synthpop)\(|pop_type['"]? *[:=] *['"]synthpops|\.ppl['"]''', 'v3 populations (including SynthPops) cannot be loaded or created; let cv.Sim create the population, or export the ages, sexes and contacts with v3 and rebuild them (see the migration rules)'],
    [sim + r'\.pars\b(?!\[)',                                    "sim.pars only has the Starsim sim parameters (e.g. n_agents); use sim['key'] for the v3 parameters, or sim.diseases.covid.pars for the COVID parameters"],
    [r'cv\.migrate\(',                                           'cv.migrate() is removed; v3 sims can be loaded to read their results and parameters, but not rerun'],
    [r'cv\.(ParsObj|BaseSim|BasePeople|Person)\b',               'The v3 base classes are removed; cv.Sim and cv.People are based on ss.Sim and ss.People'],
    [r'cv\.Result\(',                                            'cv.Result is now ss.Result, which has a different signature'],
    [r'\.(init_people|init_immunity|init_variants|init_interventions|init_analyzers|validate_pars)\(', 'v3 internals of initialization are removed; the sim is validated and initialized by sim.initialize()'],
    [r'''[(,] *version *= *['"]([01]\.|2\.0)''',         'Parameters from versions before 2.1.0 are not available; see the migration rules for how to reproduce them'],
    [r'reset_seed',                                              'reset_seed has no effect: results only depend on rand_seed, since each distribution has its own random number stream'],
    [r'''_by_variant['"]?\]\[''',                                'Results by variant now have time as the first axis, i.e. [:, variant] rather than [variant, :] (or use the variant name, e.g. .delta)'],
    [r'(people|ppl)\.(get|person_keys|dur_keys|to_arr|to_list|from_list|make_edgelist|_resize_arrays)\(', 'v3 internal with no v4 equivalent; use people.keys(), people.to_df() or people.person(uid)'],
    [r'(people|ppl)\.t *=',                                      'v3 internal; use sim.initialize(reset=True) to rerun a sim'],
    [r'self\.(t|pars|sim|dists|results) *=[^=]',                 'In an intervention or analyzer, Starsim reserves this attribute, so use a different name (e.g. self.tvec)'],
    [r'def finalize\(self\) *:',                                 'A finalize() method without the sim argument must call super().finalize()'],
    [r'\.combine\(',                                             "msim.combine() doesn't merge people; the combined sim keeps the first sim's people"],
    [r'log_scale *= *\[',                                        'Plot titles use the v4 result labels (e.g. "New infections" rather than "Number of new infections")'],
]


def migrate_findinds(line):
    '''
    Replace sc.findinds() with cv.true() where its only argument is a people array, e.g.
    ``sc.findinds(sim.people.age < 20)``. The two-argument form (``sc.findinds(arr, val)``) is left as is.
    '''
    func = 'sc.findinds('
    start = line.find(func)
    while start >= 0:
        depth = 1
        pos = start + len(func)
        has_comma = False
        while pos < len(line) and depth > 0: # Find the matching closing parenthesis
            char = line[pos]
            depth += (char in '([') - (char in ')]')
            has_comma = has_comma or (char == ',' and depth == 1)
            pos += 1
        arg = line[start+len(func):pos-1]
        if re.search(r'(people|ppl)\.', arg) and not has_comma:
            line = line[:start] + 'cv.true(' + line[start+len(func):]
        start = line.find(func, start + 1)
    return line


def migrate_code(code):
    '''
    Migrate Covasim v3 code to v4.

    Args:
        code (str): the v3 code

    Returns:
        A tuple of the migrated code, a list of the mechanical changes made, and a list of
        lines to check by hand; each entry of the lists is [line number, line, description]

    **Example**::

        new, changes, to_check = cv.migrate3to4.migrate_code('if sim.t == 10: n = len(sim.people)')
    '''
    lines = code.split('\n')
    changes = []
    to_check = []
    for i,line in enumerate(lines):
        lineno = i + 1
        for pattern, replacement, description in rules:
            new = re.sub(pattern, replacement, line)
            if new != line:
                changes.append([lineno, line.strip(), description])
                line = new
        new = migrate_findinds(line)
        if new != line:
            changes.append([lineno, line.strip(), findinds_rule])
            line = new
        lines[i] = line
        for pattern, description in checks:
            if re.search(pattern, line):
                to_check.append([lineno, line.strip(), description])
    return '\n'.join(lines), changes, to_check


def migrate_file(filename, apply=False, verbose=True):
    '''
    Migrate a file of Covasim v3 code to v4.

    Args:
        filename (str): the file to migrate: a Python script, or a Jupyter notebook (.ipynb), whose code cells are migrated
        apply (bool): whether to change the file (default: only show the changes)
        verbose (bool): whether to print the changes, and the lines to check by hand

    Returns:
        A tuple of the mechanical changes and the lines to check by hand; see migrate_code()
    '''
    text = sc.loadtext(filename)
    is_notebook = str(filename).endswith('.ipynb')
    if is_notebook: # Migrate each code cell; line numbers are within the cell
        notebook = json.loads(text)
        cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        blocks = {f'{filename} (cell {c})':''.join(cell['source']) for c,cell in enumerate(notebook['cells']) if cell['cell_type'] == 'code'}
    else:
        blocks = {str(filename):text}

    changes = []
    to_check = []
    new_blocks = []
    for label, code in blocks.items():
        new, block_changes, block_to_check = migrate_code(code)
        new_blocks.append(new)
        changes += block_changes
        to_check += block_to_check
        if verbose:
            if block_changes:
                diff = difflib.unified_diff(code.split('\n'), new.split('\n'), fromfile=label, tofile=f'{label} (v4)', lineterm='', n=0)
                print('\n'.join(diff))
            for lineno, line, description in block_to_check:
                print(f'{label}:{lineno}: CHECK: {description}\n    {line}')

    if apply and changes:
        if is_notebook:
            for cell, new in zip(cells, new_blocks):
                cell['source'] = new.splitlines(keepends=True)
            new = json.dumps(notebook, indent=1, ensure_ascii=False) + '\n' # The format Jupyter uses
        sc.savetext(filename, new)
    return changes, to_check


def migrate(paths, apply=False, verbose=True):
    '''
    Migrate Covasim v3 scripts to v4.

    Args:
        paths (str/list): the files to migrate; for folders, every .py and .ipynb file inside
        apply (bool): whether to change the files (default: only show the changes)
        verbose (bool): whether to print the changes, and the lines to check by hand

    Returns:
        A dict of the mechanical changes and the lines to check by hand, by filename

    **Example**::

        out = cv.migrate3to4.migrate('my_folder')
    '''
    filenames = []
    for path in sc.tolist(paths):
        path = sc.path(path)
        if path.is_dir():
            for fn in sorted(path.rglob('*')):
                hidden = any([part.startswith('.') for part in fn.relative_to(path).parts]) # e.g. .ipynb_checkpoints
                if fn.suffix in ['.py', '.ipynb'] and not hidden:
                    filenames.append(fn)
        else:
            filenames.append(path)

    out = {}
    for filename in filenames:
        changes, to_check = migrate_file(filename, apply=apply, verbose=verbose)
        if changes or to_check:
            out[str(filename)] = dict(changes=changes, to_check=to_check)

    if verbose:
        n_changes = sum([len(entry['changes']) for entry in out.values()])
        n_checks  = sum([len(entry['to_check']) for entry in out.values()])
        action = 'changed' if apply else 'to change (use --apply to change them)'
        print(f'\n{len(filenames)} file(s): {n_changes} line(s) {action}, {n_checks} line(s) to check by hand')
    return out


def main():
    ''' Run the migration from the command line (the ``covasim-migrate3to4`` script) '''
    parser = argparse.ArgumentParser(prog='covasim-migrate3to4', description='Migrate Covasim v3 scripts to v4')
    parser.add_argument('paths', nargs='+', help='the files to migrate; for folders, every .py and .ipynb file inside')
    parser.add_argument('--apply', action='store_true', help='change the files (default: only show the changes)')
    args = parser.parse_args()
    migrate(args.paths, apply=args.apply)
    return


if __name__ == '__main__':
    main()
