from setuptools import find_packages
import importlib.metadata as meta

PACKAGE = find_packages(
    '../..',
    exclude=['test*', 'Notebook*', 'tutorial*']
)[0]

project = PACKAGE
copyright = '2025, Georg Heimel'  # noqa: A001
author = 'Georg Heimel'
release = meta.version(PACKAGE)
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

autodoc_typehints = 'none'
add_module_names = False
autosectionlabel_prefix_document = True

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
