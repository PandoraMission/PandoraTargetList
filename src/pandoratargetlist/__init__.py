__version__ = "0.1.0"
# Standard library
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
HOMEDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/"
TARGDEFDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + '/target_definition_files/'

from .json_targets import make_json_file  # noqa
