from setuptools import find_packages
import importlib.metadata as meta

PACKAGE = find_packages('../..', exclude=['test*', 'Notebook*'])[0]

project = PACKAGE
copyright = '2024, Georg Heimel'
author = 'Georg Heimel'
version = meta.version(PACKAGE)
release = version

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

autodoc_typehints = 'none'
add_module_names = False

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']