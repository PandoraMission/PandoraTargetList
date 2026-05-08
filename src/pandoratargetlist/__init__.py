__version__ = "1.0.1"
# Standard library
import os  # noqa

import pandorapsf as ppsf  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
HOMEDIR = os.sep.join(PACKAGEDIR.split(os.sep)[:-2]) + os.sep
TARGDEFDIR = os.sep.join(PACKAGEDIR.split(os.sep)[:-2]) + os.sep + "target_definition_files" + os.sep

from .targets import Target  # noqa
