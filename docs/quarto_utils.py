"""
Utilities for processing Covasim docs (notebooks and API reference).
"""

import os
import sys
import importlib
import subprocess
import sciris as sc
import covasim as cv

default_folders = ['tutorials']
temp_patterns = ['**/my-*.*']
temp_items = []

timeout = 600
yay = '✓'
boo = '😢'


def run(cmd):
    """Verbose version of subprocess.run"""
    sc.printgreen(f'\n> {cmd}\n')
    return subprocess.run(cmd, check=True, shell=True)


@sc.timer('Update version')
def update_version(pkg=cv):
    sc.heading('Updating docs version number...')
    filename = '_variables.yml'
    data = dict(version=cv.__version__, versiondate=cv.__versiondate__)
    orig = sc.loadyaml(filename)
    if data != orig:
        sc.saveyaml(filename, data)
        print('Version updated to:', data)
    else:
        print('Version already correct:', orig)
    return


@sc.timer('Build API docs')
def build_api_docs():
    sc.heading('Building API documentation...')
    run('python -m quartodoc build --config _quarto.yml')
    return


@sc.timer('Customize aliases')
def customize_aliases(mod_name='covasim', json_path='objects.json'):
    """Add top-level aliases (e.g. covasim.Sim) to the objects inventory."""
    sc.heading('Customizing aliases ...')
    mod = importlib.import_module(mod_name)
    mod_items = dir(mod)

    json = sc.loadjson(json_path)
    items = json['items']
    names = [item['name'] for item in items]
    print(f'  Loaded {len(json["items"])} items')

    dups = []
    for item in items:
        parts = item['name'].split('.')
        if len(parts) < 3 or parts[0] != mod_name:
            continue
        objname = parts[2]
        if objname in mod_items:
            remainder = '.'.join(parts[2:])
            alias = f'{mod_name}.{remainder}'
            if alias not in names:
                dup = sc.dcp(item)
                dup['name'] = alias
                dups.append(dup)

    items.extend(dups)
    sc.savejson(json_path, json)
    print(f'  Saved {len(json["items"])} items')
    return


@sc.timer('Build interlinks')
def build_interlinks():
    sc.heading('Building docs links...')
    return run('python -m quartodoc interlinks')


@sc.timer('Build objects.inv')
def build_objects_inv(json_path='objects.json', inv_path='objects.inv'):
    """Convert quartodoc JSON inventory to Sphinx-compatible objects.inv."""
    import sphobjinv as soi
    sc.heading('Building Sphinx objects.inv ...')
    data = sc.loadjson(json_path)
    inv = soi.Inventory()
    inv.project = data.get('project', 'covasim')
    inv.version = str(data.get('version', cv.__version__))
    for item in data['items']:
        inv.objects.append(soi.DataObjStr(
            name=item['name'],
            domain=item['domain'],
            role=item['role'],
            priority=str(item.get('priority', '1')),
            uri=item['uri'],
            dispname=item.get('dispname', '-') or '-',
        ))
    with open(inv_path, 'wb') as f:
        f.write(soi.compress(inv.data_file()))
    print(f'  Wrote {len(inv.objects)} entries to {inv_path}')
    return


def qmd2py(qmd_path, py_path=None, keep_text=True):
    """Convert a .qmd file to a .py file by extracting Python code cells."""
    qmd_path = sc.path(qmd_path)
    if py_path is None:
        py_path = qmd_path.with_suffix('.py')
    else:
        py_path = sc.path(py_path)

    text = sc.loadtext(qmd_path)
    lines = text.splitlines()

    chunks = []
    in_block = False
    current_cell = []
    current_text = []
    block_start_line = None

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith('```{python}'):
            if in_block:
                raise ValueError(f'Nested or unclosed code block at line {i}')
            if keep_text and current_text:
                chunks.append(('text', current_text))
                current_text = []
            in_block = True
            block_start_line = i
            current_cell = []
        elif stripped == '```' and in_block:
            chunks.append(('code', current_cell))
            in_block = False
            current_cell = []
            block_start_line = None
        elif in_block:
            current_cell.append(line)
        elif keep_text:
            current_text.append(line)

    if in_block:
        raise ValueError(f'Unclosed code block starting at line {block_start_line}')

    if keep_text and current_text:
        chunks.append(('text', current_text))

    parts = []
    cell_num = 0
    for kind, content in chunks:
        if kind == 'code':
            cell_num += 1
            processed = []
            for line in content:
                if line.lstrip().startswith(('%', '!')):
                    processed.append(f'# {line}  # IPython not supported in Python files')
                else:
                    processed.append(line)
            parts.append(f'#%% Cell {cell_num}\n' + '\n'.join(processed))
        else:
            commented = '\n'.join(f'# {line}' if line.strip() else '#' for line in content)
            parts.append(commented)

    output = '\n\n\n'.join(parts) + '\n'
    sc.savetext(py_path, output)
    return py_path


def execute_notebook(path, tidy=True):
    """Execute a single Jupyter notebook and return success/failure."""
    nb_path = path.name
    os.chdir(path.parent)
    with sc.timer(label=sc.ansi.green(f'    Execution time for {nb_path}')) as T:
        base = nb_path.removesuffix('.ipynb')
        py_path = base + '.py'
        try:
            print(f'Converting {nb_path} to {py_path}...')
            sc.loadtext(nb_path)  # ensure readable
            env = {**os.environ, 'MPLBACKEND': 'agg', 'COVASIM_VERBOSE': '0'}
            subprocess.run(['jupyter', 'nbconvert', '--to', 'python', nb_path], check=True, capture_output=True, cwd=path.parent)
            print(f'Executing {py_path}...')
            subprocess.run(['python', py_path], check=True, capture_output=True, cwd=path.parent, env=env)
            string = f'{yay} {base} executed successfully '
        except subprocess.CalledProcessError as e:
            string = f'{boo} Execution failed for {base}: {e}\n'
        except Exception as e:
            string = f'{boo} Error processing {base}: {str(e)}\n'
        finally:
            if tidy:
                sc.rmpath(py_path, die=False)

    string += f'(time: {T.total:0.1f} s)'
    print(string)
    return string


@sc.timer('Execute notebooks')
def execute_notebooks(*args, folders=None, tidy=True, debug=False):
    """Execute notebooks in parallel and print which succeeded / failed."""
    T = sc.timer()
    cwd = sc.thispath(__file__)
    results = sc.objdict()
    string = ''

    if args:
        notebooks = [sc.path(notebook).resolve() for notebook in args]
    else:
        notebooks = []
        folders = sc.ifelse(folders, default_folders)
        for folder in folders:
            folder_path = cwd / folder
            notebooks += [folder_path / f for f in sc.getfilepaths(folder_path, '*.ipynb')]

    def execute(i, path, pause=1.0):
        delay = i * pause
        sc.timedsleep(delay)
        return execute_notebook(path, tidy=tidy)

    sc.heading(f'Running {len(notebooks)} notebooks...')
    notebook_list = list(enumerate(notebooks))
    out = sc.parallelize(execute, notebook_list, maxcpu=0.9, interval=1.0, lbkwargs=dict(verbose=False), serial=debug)
    string += sc.strjoin(out, sep=f'\n\n\n{"—"*90}\n')
    for nb, res in zip(notebooks, out):
        results[str(nb)] = res

    sc.heading('Results')
    print(string)

    sc.heading('Summary')
    n_yay = string.count(yay)
    n_boo = string.count(boo)
    summary = f'{n_yay} succeeded, {n_boo} failed\n'

    results.sort('values')
    for nb, res in results.items():
        timestr = res.split('time: ')[1][:-1]
        suffix = f'{sc.path(nb).name:30s} ({timestr})'
        if yay in res:
            summary += f'\n{sc.ansi.green("Succeeded")}: {suffix}'
        elif boo in res:
            summary += f'\n{sc.ansi.red("   Failed")}: {suffix}'
    print(summary)

    T.toc()
    return results


@sc.timer('Clean outputs')
def clean_outputs(folders=None, sleep=3, patterns=None):
    """Clear temporary notebook outputs and artifacts."""
    sc.heading('Cleaning outputs ...')
    if folders is None:
        folders = default_folders
    if patterns is None:
        patterns = temp_patterns
    filenames = sc.dcp(temp_items)
    for pattern in patterns:
        for folder in folders:
            filenames += sc.getfilelist(folder=folder, pattern=pattern, recursive=True)
    if len(filenames):
        print(f'Deleting: {sc.newlinejoin(filenames)}\nin {sleep} seconds')
        sc.timedsleep(sleep)
        for filename in filenames:
            sc.rmpath(filename, verbose=True, die=False)
    else:
        print('No files found to clean')
    return


if __name__ == '__main__':

    if 'pre' in sys.argv:
        sc.heading('Starting Quarto docs build', divider='★')
        update_version()
        build_api_docs()
        customize_aliases()
        build_interlinks()
        build_objects_inv()

    elif 'post' in sys.argv:
        clean_outputs()

    elif len(sys.argv) > 1:
        errormsg = f'Argument must be "pre" or "post", not {sys.argv}'
        raise ValueError(errormsg)
