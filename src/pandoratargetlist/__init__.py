__version__ = "0.1.3"
# Standard library
import os  # noqa

import pandorapsf as ppsf  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
HOMEDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/"
TARGDEFDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/target_definition_files/"

VDA_PSF = ppsf.PSF.from_name("VISDA")
NIRDA_PSF = ppsf.PSF.from_name("NIRDA")

from .targets import make_target_file  # noqa
