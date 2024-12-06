# Standard library
import json
import os

# First-party/Local
from pandoratargetlist import TARGDEFDIR


# Validate the formatting of the target definition files
def test_file_format():
    cats = [cat for cat in os.listdir(TARGDEFDIR) if '.' not in cat]
    for cat in cats:
        catpath = TARGDEFDIR + cat + '/'
        filelist = [f for f in os.listdir(catpath) if f[-5:] == '.json']
        # targlist = [targ.split('_target')[0] for targ in filelist]
        for file in filelist:
            with open(catpath + file, 'r') as f:
                data = json.load(f)
            assert 