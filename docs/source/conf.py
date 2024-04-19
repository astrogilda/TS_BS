from datetime import datetime

# sys.path.insert(0, str(Path("../").resolve()))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tsbootstrap"
current_year = datetime.now().year
copyright = f"2023 - {current_year} (MIT License), Sankalp Gilda"
author = "Sankalp Gilda"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []
suppress_warnings = ["ref.undefined", "ref.footnote"]

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#module-sphinx.ext.intersphinx
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "arch": ("https://arch.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_rtd_theme"
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 59afbe7 (debugging icon_links issue with sphinx build)
=======
>>>>>>> 840379a (using sphinx_rtd_theme instead of furo)
=======
=======
>>>>>>> 59afbe7 (debugging icon_links issue with sphinx build)
>>>>>>> 3094ea9 (debugging icon_links issue with sphinx build)
=======

>>>>>>> c9b4cb4 (conditionally removed testing on windows python 3.12)
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "navigation_depth": 3,
    "navigation_with_keys": False,
}

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 3094ea9 (debugging icon_links issue with sphinx build)
=======
>>>>>>> 8755079 (using sphinx_rtd_theme instead of furo)
=======
>>>>>>> 59afbe7 (debugging icon_links issue with sphinx build)
<<<<<<< HEAD
=======
=======
>>>>>>> 8755079 (using sphinx_rtd_theme instead of furo)
>>>>>>> 840379a (using sphinx_rtd_theme instead of furo)
=======
>>>>>>> 3094ea9 (debugging icon_links issue with sphinx build)
=======

>>>>>>> c9b4cb4 (conditionally removed testing on windows python 3.12)
# html_theme = "furo"
html_static_path = []
