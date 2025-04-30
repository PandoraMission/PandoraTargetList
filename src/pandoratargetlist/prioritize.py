# Functions to prioritize targets and generate the priority files

import os

import pandas as pd
import numpy as np
from astropy.time import Time

from pandoratargetlist import TARGDEFDIR, __version__


# Make priority files

# Update priority files


def make_priorities(
    dirs=[],
    author='system'
):
    """
    Function to make priority file.

    Parameters
    ----------

    Returns
    -------
    """
    # Set observation time per category
    time_req_dict = {
        "primary-exoplanet": 10,
        "secondary-exoplanet": 10,
        "auxiliary-exoplanet": 4,
        "auxiliary-standard": 120,
        "monitoring-standard": 1000,
        "occultation-standard": 600,
    }

    # Maybe something like time_req_dict.update() with user-params inside the parenthesis?

    if len(dirs) == 0:
        targ_dirs = [
            x
            for x in os.listdir(TARGDEFDIR)
            if os.path.isdir(os.path.join(TARGDEFDIR, x))
        ]
    else:
        targ_dirs = dirs

    for dir in targ_dirs:
        # Detect targs in directory
        dirpath = TARGDEFDIR + dir + "/"
        filelist = [f for f in os.listdir(dirpath) if f[-5:] == ".json"]
        targs = [targ.split("_target")[0] for targ in filelist]

        # Define priority file path
        priorityfile = dirpath + dir + "_priorities.csv"
        tmp_priorityfile = dirpath + dir + "_priorities_tmp.csv"

        # Delete temp priority file if it exists
        if os.path.isfile(tmp_priorityfile):
            os.remove(tmp_priorityfile)

        if dir == "occultation-standard":
            targs_planet = [targ for targ in targs if "DR3" not in targ]
            targs_star = [targ for targ in targs if "DR3" in targ]
            targs = targs_planet + targs_star

        # Define priority space
        priorities = np.linspace(0.9, 0.2, len(targs))

        # Make observing time lists
        time_req = np.ones(len(targs)) * time_req_dict[dir]

        if dir == "occultation-standard":
            gaia_flag = ["DR3" in s for s in targs]
            time_req[gaia_flag] = 1000

        time_obs = np.zeros(len(targs))
        time_rem = time_req - time_obs

        # Open file and setup standard header
        f = open(tmp_priorityfile, "a")
        f.write("# Priority file for " + dir + "\n")
        f.write("# Version " + __version__ + "\n")
        f.write("# Updated on: " + Time.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("# Updated by: " + author + "\n")
        f.write("#\n")
        f.write("# Column 1: Rank\n")
        f.write("# Column 2: Target Name\n")
        f.write("# Column 3: Priority\n")

        # Make priorities dataframe
        df_priorities = pd.DataFrame(
            {
                "target": targs,
                "priority": priorities,
            }
        )

        if "exoplanet" in dir:
            df_priorities["transits_req"] = time_req.astype(int)
            df_priorities["transits_obs"] = time_obs.astype(int)
            df_priorities["transits_rem"] = time_rem.astype(int)

            f.write("# Column 4: Number of Transits Requested\n")
            f.write("# Column 5: Number of Transits Observed\n")
            f.write("# Column 6: Number of Transits Remaining\n")

        else:
            df_priorities["hours_req"] = time_req
            df_priorities["hours_obs"] = time_obs
            df_priorities["hours_rem"] = time_rem

            f.write("# Column 4: Number of Hours Requested\n")
            f.write("# Column 5: Number of Hours Observed\n")
            f.write("# Column 6: Number of Hours Remaining\n")

        print(df_priorities)

        # Save priority file
        df_priorities.to_csv(f, index=True, index_label="rank")
        f.close()

        # Rename temp priority file to actual priority file
        os.remove(priorityfile)
        os.rename(tmp_priorityfile, priorityfile)


# Funtion to recalculate priorities
def _recalc_priorities():
    """Function to recalculate priorities after a new target has been added"""
    print("Work in progress!")
    return None
