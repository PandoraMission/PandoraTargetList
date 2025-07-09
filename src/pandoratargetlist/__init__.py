__version__ = "0.2.0"
# Standard library
import os  # noqa

import pandorapsf as ppsf  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
HOMEDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/"
TARGDEFDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/target_definition_files/"

from .targets import Target  # noqa
