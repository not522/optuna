# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import pkg_resources

import plotly.io as pio
from sphinx_gallery.sorting import FileNameSortKey

nitpicky = True
nitpick_ignore = [
    ('py:class', 'go.Figure'),
    ('py:class', 'plotly.graph_objs.Figure'),
    ('py:class', 'Axes'),
    ('py:class', 'matplotlib.axes.Axes'),
    ('py:class', 'torch.Tensor'),
    ('py:class', 'numpy.ndarray'),
    ('py:class', 'pandas.core.series.Series'),
    ('py:class', 'scipy.sparse.base.spmatrix'),
    ('py:class', '..'),
    ('py:class', 'GradientDescentTrainer'),
    ('py:class', 'pd.DataFrame'),
    ('py:class', 'pd.Series'),
    ('py:class', 'default=True'),
    ('py:class', 'score'),
    ('py:class', 'sklearn.pipeline.Pipeline'),
    ('py:class', 'estimator instance'),
    ('py:class', 'lgb.Booster'),
    ('py:class', 'lgb.CVBooster'),
    ('py:class', 'optuna.distributions.BaseDistribution'),
    ('py:class', 'optuna.storages.BaseStorage'),
    ('py:class', 'optuna.storages._base.BaseStorage'),
    ('py:class', 'optuna.pruners._base.BasePruner'),
    ('py:class', 'optuna.importance._base.BaseImportanceEvaluator'),
    ('py:attr', 'optuna.study.Study.user_attr'),
    ('py:func', 'optuna.study.Study.get_trial'),
    ('py:class', 'optuna.importance._fanova._fanova._Fanova'),
    ('py:class', 'optuna.integration.chainermn.ChainerMNStudy'),
    ('py:class', 'ChainerMNTrial'),
    ('py:class', 'CommunicatorBase'),
    ('py:class', 'ObjectiveFuncType'),
    ('py:class', 'optuna.study.study.Study'),
    ('py:class', 'optuna.Study'),
    ('py:class', 'optuna.trial._trial.Trial'),
    ('py:class', 'optuna.importance.BaseImportanceEvaluator'),
    ('py:class', 'optuna.multi_objective.trial.Trial'),
    ('py:class', 'multi_objective.samplers.BaseMultiObjectiveSampler'),
    ('py:obj', 'optuna.integration.FastAIV1PruningCallback.on_epoch_end'),
    ('py:obj', 'optuna.integration.FastAIV2PruningCallback.after_epoch'),
    ('py:obj', 'optuna.integration.FastAIV2PruningCallback.after_fit'),
    ('py:obj', 'optuna.integration.KerasPruningCallback.on_epoch_end'),
    ('py:obj', 'optuna.integration.PyTorchLightningPruningCallback.on_fit_end'),
    ('py:obj', 'optuna.integration.PyTorchLightningPruningCallback.on_init_start'),
    ('py:obj', 'optuna.integration.PyTorchLightningPruningCallback.on_validation_end'),
    ('py:obj', 'optuna.integration.SkorchPruningCallback.on_epoch_end'),
    ('py:obj', 'optuna.integration.TFKerasPruningCallback.on_epoch_end'),
    ('py:obj', 'optuna.integration.TensorFlowPruningHook.after_run'),
    ('py:obj', 'optuna.integration.TensorFlowPruningHook.before_run'),
    ('py:obj', 'optuna.integration.TensorFlowPruningHook.begin'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.report'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.set_system_attr'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.set_user_attr'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.should_prune'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.suggest_categorical'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.suggest_float'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.suggest_int'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.datetime_start'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.distributions'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.number'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.params'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.system_attrs'),
    ('py:obj', 'optuna.integration.TorchDistributedTrial.user_attrs'),
    ('py:func', 'optuna.integration.TorchDistributedTrial.suggest_float'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTuner.compare_validation_metrics'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTuner.higher_is_better'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTuner.tune_bagging'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTuner.tune_feature_fraction'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTuner.tune_feature_fraction_stage2'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTuner.tune_min_data_in_leaf'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTuner.tune_num_leaves'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTuner.tune_regularization_factors'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTunerCV.compare_validation_metrics'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTunerCV.higher_is_better'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTunerCV.tune_bagging'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTunerCV.tune_feature_fraction'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTunerCV.tune_feature_fraction_stage2'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTunerCV.tune_min_data_in_leaf'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTunerCV.tune_num_leaves'),
    ('py:obj', 'optuna.integration.lightgbm.LightGBMTunerCV.tune_regularization_factors'),
    ('py:obj', '(TrialState.COMPLETE,)'),
    ('py:obj', 'optuna.trial.FixedTrial.report'),
    ('py:obj', 'optuna.trial.FixedTrial.set_system_attr'),
    ('py:obj', 'optuna.trial.FixedTrial.set_user_attr'),
    ('py:obj', 'optuna.trial.FixedTrial.should_prune'),
    ('py:obj', 'optuna.trial.FixedTrial.suggest_categorical'),
    ('py:obj', 'optuna.trial.FixedTrial.suggest_float'),
    ('py:obj', 'optuna.trial.FixedTrial.suggest_int'),
    ('py:obj', 'optuna.trial.FixedTrial.datetime_start'),
    ('py:obj', 'optuna.trial.FixedTrial.distributions'),
    ('py:obj', 'optuna.trial.FixedTrial.number'),
    ('py:obj', 'optuna.trial.FixedTrial.params'),
    ('py:obj', 'optuna.trial.FixedTrial.system_attrs'),
    ('py:obj', 'optuna.trial.FixedTrial.user_attrs'),
    ('py:obj', 'optuna.trial.FrozenTrial.set_system_attr'),
    ('py:obj', 'optuna.trial.FrozenTrial.set_user_attr'),
    ('py:obj', 'optuna.trial.FrozenTrial.suggest_categorical'),
    ('py:obj', 'optuna.trial.FrozenTrial.suggest_float'),
    ('py:obj', 'optuna.trial.FrozenTrial.suggest_int'),
    ('py:obj', 'optuna.trial.FrozenTrial.system_attrs'),
    ('py:func', 'optuna.trial.FrozenTrial.suggest_float'),
    ('py:meth', 'evaluate'),
    ('py:class', 'optuna.integration.fastaiv2.FastAIV2PruningCallback'),
    ('py:func', 'optuna.multi_objective.samplers.MultiObjectiveBaseSampler.sample_relative'),
    ('py:meth', 'optuna.integration.LightGBMTuner.get_best_booster'),
    ('py:meth', 'optuna.integration.lightgbm.LightGBMTuner.__init__'),
    ('py:meth', 'optuna.integration.LightGBMTunerCV.get_best_booster'),
    ('py:meth', 'optuna.integration.lightgbm.LightGBMTunerCV.__init__'),
    ('py:func', 'optuna.logging.enable_propogation'),
    ('py:meth', 'optuna.storages.BaseStorage.get_heartbeat_interval'),
    ('py:func', 'optuna.trial.FixedTrial.suggest_float'),
    ('py:func', 'optuna.trial.FrozenTrial.set_user_attr'),
    ('py:obj', 'optuna.multi_objective.trial.FrozenMultiObjectiveTrial.last_step'),
    ('py:obj', 'optuna.multi_objective.trial.FrozenMultiObjectiveTrial.system_attrs'),
]
autodoc_type_aliases = {
    'Study': 'optuna.study.Study',
    'optuna.Study': 'optuna.study.Study',
    'Trial': 'optuna.trial.Trial',
    'trial_module.Trial': 'optuna.trial.Trial',
    'FrozenTrial': 'optuna.trial.FrozenTrial',
    'StudyDirection': 'optuna.study.StudyDirection',
    'StudySummary': 'optuna.study.StudySummary',
    'TrialState': 'optuna.trial.TrialState',
}

__version__ = pkg_resources.get_distribution("optuna").version

# -- Project information -----------------------------------------------------

project = "Optuna"
copyright = "2018, Optuna Contributors."
author = "Optuna Contributors."

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "cliff.sphinxext",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_plotly_directive",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {"logo_only": True, "navigation_with_keys": True}

html_favicon = "../image/favicon.ico"

html_logo = "../image/optuna-logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "Optunadoc"

# -- Options for LaTeX output ------------------------------------------------

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
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "Optuna.tex", "Optuna Documentation", "Optuna Contributors.", "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "optuna", "Optuna Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Optuna",
        "Optuna Documentation",
        author,
        "Optuna",
        "One line description of project.",
        "Miscellaneous",
    ),
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# -- Extension configuration -------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "exclude-members": "with_traceback",
}

# Sphinx Gallery
pio.renderers.default = "sphinx_gallery"

sphinx_gallery_conf = {
    "examples_dirs": [
        "../../tutorial",
    ],
    "gallery_dirs": [
        "tutorial",
    ],
    "within_subsection_order": FileNameSortKey,
    "filename_pattern": r"/*\.py",
    "first_notebook_cell": None,
}

# matplotlib plot directive
plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False

# sphinx plotly directive
plotly_include_source = True
plotly_formats = ["html"]
plotly_html_show_formats = False
plotly_html_show_source_link = False
