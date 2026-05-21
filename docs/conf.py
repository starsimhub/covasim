# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import sciris as sc
import covasim as cv

# Set environment
os.environ['SPHINX_BUILD'] = 'True' # This is used so cv.options.set('jupyter') doesn't reset the Matplotlib renderer
on_rtd = os.environ.get('READTHEDOCS') == 'True'


# -- Project information -----------------------------------------------------

project = 'Covasim'
copyright = f'2020 - {sc.now().year}, Gates Foundation. All rights reserved.\nThese docs were built for Covasim version {cv.__version__}.\n'
author = 'Institute for Disease Modeling'

# The short X.Y version
version = cv.__version__

# The full version, including alpha/beta/rc tags
release = cv.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc -- causes warnings with Napoleon however
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "sphinx_design", # Add e.g. grid layout
    'sphinx_search.extension', # search across multiple docsets in domain
    "nbsphinx",
]

# Use Google docstrings
napoleon_google_docstring = True

# Configure autosummary
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_ignore_module_all = False # Respect __all__
autodoc_member_order = 'bysource' # Keep original ordering
add_module_names = False  # NB, does not work
autodoc_inherit_docstrings = False # Stops sublcasses from including docs from parent classes

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Syntax highlighting style
pygments_style = "sphinx"
modindex_common_prefix = ["covasim."]

# List of patterns, relative to source directory, to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Suppress certain warnings
suppress_warnings = ['autosectionlabel.*']


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "show_prev_next": True,
    "icon_links": [
        {"name": "Web", "url": "https://covasim.org", "icon": "fas fa-home"},
        {
            "name": "GitHub",
            "url": "https://github.com/institutefordiseasemodeling/covasim",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["navbar-side"],
    "header_links_before_dropdown": 5,
    "footer_start": ["copyright", "footer_start"],
    "footer_end": ["theme-version", "footer_end"],
}
html_sidebars = {
    "**": ["sidebar-nav-bs", "page-toc"],
}
html_logo = "images/idm-logo-transparent.png"
html_favicon = "images/favicon.ico"
html_static_path = ['_static']
html_baseurl = "https://docs.idmod.org/projects/covasim/en/latest"
html_context = {
    'rtd_url': 'https://docs.idmod.org/projects/covasim/en/latest',
    "versions_dropdown": {
        "latest": "devel (latest)",
        "stable": "current (stable)",
    },
    "default_mode": "light",
}
# Add any extra paths that contain custom files
if not on_rtd:
    html_extra_path = ['robots.txt']


# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_last_updated_fmt = '%Y-%b-%d'
html_show_sourcelink = True
html_show_sphinx = False
html_copy_source = False
htmlhelp_basename = 'Covasim'

# Add customizations
def setup(app):
    app.add_css_file("theme_overrides.css")


# Modify this to not rerun the Jupyter notebook cells -- usually set by build_docs
nb_ex_default = ['auto', 'never'][0]
nb_ex = os.getenv('NBSPHINX_EXECUTE')
if not nb_ex: nb_ex = nb_ex_default
print(f'\n\nBuilding Jupyter notebooks with build option: {nb_ex}\n\n')
nbsphinx_execute = nb_ex

# OpenSearch options
html_use_opensearch = 'docs.idmod.org/projects/covasim/en/latest'

# -- RTD Sphinx search for searching across the entire domain, default child -------------
if os.environ.get('READTHEDOCS') == 'True':

    search_project_parent = "institute-for-disease-modeling-idm"
    search_project = os.environ["READTHEDOCS_PROJECT"]
    search_version = os.environ["READTHEDOCS_VERSION"]

    rtd_sphinx_search_default_filter = f"subprojects:{search_project}/{search_version}"

    rtd_sphinx_search_filters = {
        "Search this project": f"project:{search_project}/{search_version}",
        "Search all IDM docs": f"subprojects:{search_project_parent}/{search_version}",
    }
