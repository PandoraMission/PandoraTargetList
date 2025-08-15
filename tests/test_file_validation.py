# Standard library
import json
import os

# First-party/Local
from pandoratargetlist import TARGDEFDIR


# Validate the formatting of the target definition files
def test_name_format():
    cats = [cat for cat in os.listdir(TARGDEFDIR) if "." not in cat]
    for cat in cats:
        catpath = TARGDEFDIR + cat + "/"
        filelist = [f for f in os.listdir(catpath) if f[-5:] == ".json"]
        targ_names = []
        for file in filelist:
            with open(catpath + file, "r") as f:
                data = json.load(f)
            targ_names.append(data["Star Name"])
            if "Planet Name" in data.keys():
                targ_names.append(data["Planet Name"])
        res = True
        for i in targ_names:
            if i.count(" ") > 0:
                res = False
                break
        assert res is True

        res = True
        for s in targ_names:
            if len(s) > 20:
                res = False
                break
        assert res is True
