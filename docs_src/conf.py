# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'AddingDoublingRT'
copyright = '2026, Daniel Kitzmann'
author = 'Daniel Kitzmann'
release = '1.0'
version = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Number figures, tables and code blocks, and cross-reference them with :numref:.
numfig = True

# Default language for highlighting fenced/literal code blocks.
highlight_language = 'cpp'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'AddingDoublingRT Documentation'

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'titles_only': False,
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
''',
}

latex_documents = [
    ('index', 'AddingDoublingRT.tex', 'AddingDoublingRT Documentation',
     'Daniel Kitzmann', 'manual'),
]

# -- MathJax macros (shared across all pages) --------------------------------
mathjax3_config = {
    'tex': {
        'macros': {
            'mat': [r'\mathbf{#1}', 1],
            'vect': [r'\boldsymbol{#1}', 1],
            'dd': r'\mathrm{d}',
            'ee': r'\mathrm{e}',
            'RR': r'\mathbf{R}',
            'TT': r'\mathbf{T}',
        }
    }
}
