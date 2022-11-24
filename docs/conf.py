#!/usr/bin/env python
#
# pycmtensor documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import shutil
import sys

import sphinx

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "PyCMTensor"
copyright = "2022, PyCMTensor Development Team"
author = "Melvin Wong"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = "1.3.1"
# The full version, including alpha/beta/rc tags.
release = "1.3.1"

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    # "myst_parser",
    "myst_nb",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# The master toctree document.
master_doc = "index"

suppress_warnings = ["autoapi"]

# -- Options for myst-nb -----------------------------------------------------

myst_url_schemes = ("http", "https", "mailto")
nb_execution_mode = "off"
nb_execution_allow_errors = True
nb_number_source_lines = True
myst_render_markdown_format = "myst"
# -- Options for myst-parser -------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# -- Options for autoapi -----------------------------------------------------


def skip_config_classes(app, what, name, obj, skip, options):
    if what == "data":
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_config_classes)


autoapi_add_toctree_entry = True
autoapi_python_class_content = "both"
autoapi_type = "python"
autoapi_keep_files = True
autoapi_dirs = ["../pycmtensor"]
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    # "show-inheritance-diagram",
    # "show-module-summary",
    # "special-members",
    # "imported-members",
]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "repository_url": "https://github.com/mwong009/pycmtensor",
    "use_repository_button": True,
    "repository_branch": "Master",
    "use_issues_button": True,
    "path_to_docs": "docs/",
    "home_page_in_toc": False,
    "use_edit_page_button": True,
    "show_toc_level": 1,
}

# html_logo = "path/to/myimage.png"
# html_title = "My site title"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pycmtensordoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, "pycmtensor.tex", "PyCMTensor Documentation", "Melvin Wong", "manual"),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pycmtensor", "PyCMTensor Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "pycmtensor",
        "PyCMTensor Documentation",
        author,
        "pycmtensor",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# myst_commonmark_only = True
