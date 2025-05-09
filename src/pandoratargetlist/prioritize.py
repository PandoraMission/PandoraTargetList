# Functions to prioritize targets and generate the priority files

import os
import csv

import pandas as pd
import numpy as np
from astropy.time import Time

from pandoratargetlist import TARGDEFDIR, __version__


class Priorities:
    """Class to hold and manipulate the prioritization files for the target definition files."""

    def __init__(self, category, author="system"):
        self.category = category
        self.dirpath = TARGDEFDIR + self.category + "/"
        self.priority_file = self.dirpath + self.category + "_priorities.csv"
        self.author = author
        self.is_exoplanet = "exoplanet" in self.category
        self.json_targets = self._load_json_targets()
        self.targets = []
        self._load_or_initialize_priorities()

    def _load_json_targets(self):
        """Function for determing which targets are in a category."""
        return sorted(f.stem for f in self.dirpath.glob("*_target_definition.json"))

    def _load_or_initialize_priorities(self):
        """Function that loads an existing prioritization file if it exists. Otherwise makes one."""
        if self.priority_file.exists():
            self._read_prioritization_file()
        else:
            self.targets = [
                {
                    "target": t,
                    "rank": i,
                    "priority": 0.0,
                    self._req_key(): self._default_required(t),
                    self._obs_key(): 0,
                    self._rem_key(): self._default_required(t),
                }
                for i, t in enumerate(self.json_targets)
            ]
        self._sync_targets()
        self._recalculate_priorities()

    def _req_key(self):
        return "hours_req" if self.is_exoplanet else "transits_req"

    def _obs_key(self):
        return "hours_obs" if self.is_exoplanet else "transit_obs"

    def _rem_key(self):
        return "hours_rem" if self.is_exoplanet else "transits_rem"

    def _default_required(self, target):
        """Sets default observation amounts for each category."""
        name = self.category
        if name == "primary-exoplanet":
            return 10
        if name == "secondary-exoplanet":
            return 10
        if name == "auxiliary-exoplanet":
            return 4
        if name == "auxiliary-standard":
            return 120
        if name == "monitoring-standard":
            return 1000
        if name == "occultation-standard":
            return 1000 if "DR3" in target else 600
        return 0

    def _read_prioritization_file(self):
        """Function to read in a prioritization file."""
        with open(self.priority_file, newline="") as f:
            reader = csv.DictReader((line for line in f if not line.startswith("#")))
            self.targets = [
                {
                    "target": row["target"],
                    "rank": int(row["rank"]),
                    "priority": float(row["priority"]),
                    self._req_key(): float(row[self._req_key()]),
                    self._obs_key(): float(row[self._obs_key()]),
                    self._rem_key(): float(row[self._rem_key()]),
                }
                for row in reader
            ]

    def _sync_targets(self):
        """Function to add/remove targets based on whether they're in the category."""
        existing = {t["target"]: t for t in self.targets}
        updated = []

        for name in self.json_targets:
            if name in existing:
                updated.append(existing[name])
            else:
                updated.append(
                    {
                        "target": name,
                        "rank": -1,
                        "priority": 0.0,
                        self._req_key(): self._default_required(name),
                        self._obs_key(): 0,
                        self._rem_key(): self._default_required(name),
                    }
                )

        updated.sort(key=lambda x: x["rank"] if x["rank"] >= 0 else float("inf"))
        for i, t in enumerate(updated):
            t["rank"] = i
        self.targets = updated

    def _recalculate_priorities(self):
        """Function to generically recalculate priorities."""
        if self.category == "occultation-standard":
            self.targets.sort(key=lambda x: ("DR3" in x["target"], x["rank"]))

        self.targets.sort(key=lambda x: x["priority"], reverse=True)

        n = len(self.targets)
        for i, t in enumerate(self.targets):
            t["rank"] = i
            t["priority"] = round(0.9 - (0.6 * i / max(n - 1, 1)), 4)

    def _generate_header(self):
        """Function to generate the header for the output priorities file."""
        now = Time.now().strftime("%Y-%m-%d %H:%M:%S")
        header_lines = [
            f"# Prioritization file for {self.category}",
            f"# Version: {__version__}",
            f"# Updated by: {self.author}",
            f"# Updated on: {now}",
            "#",
            "# Column 1: Rank",
            "# Column 2: Target Name",
            "# Column 3: Priority",
        ]
        if self.is_exoplanet:
            other_lines = [
                "# Column 4: Number of Transits Requested",
                "# Column 5: Number of Transits Observed",
                "# Column 6: Number of Transits Remaining",
            ]
            header_lines = header_lines + other_lines
        else:
            other_lines = [
                "# Column 4: Number of Hours Requested",
                "# Column 5: Number of Hours Observed",
                "# Column 6: Number of Hours Remaining",
            ]
            header_lines = header_lines + other_lines
        return "\n".join(header_lines) + "\n"

    def move_target(
        self, target_name, *, new_rank=None, delta=None, above=None, below=None
    ):
        """Function to move a target to a specific rank or proximity to another target."""
        index = next(
            (i for i, t in enumerate(self.targets) if t["target"] == target_name), None
        )
        if index is None:
            raise ValueError(f"Target {target_name} not found.")

        target = self.targets.pop(index)

        if new_rank is not None:
            new_rank = max(0, min(new_rank, len(self.targets)))
        elif delta is not None:
            new_rank = max(0, min(index + delta, len(self.targets)))
        elif above:
            new_rank = next(
                i for i, t in enumerate(self.targets) if t["target"] == above
            )
        elif below:
            new_rank = (
                next(i for i, t in enumerate(self.targets) if t["target"] == below) + 1
            )
        else:
            raise ValueError("Must specify new_rank, delta, above, or below")

        self.targets.insert(new_rank, target)
        self._recalculate_priorities()

    def set_priority(self, target_name, priority):
        """Function to manually set the priority of a target."""
        for t in self.targets:
            if t["target"] == target_name:
                t["priority"] = float(priority)
                return
        raise ValueError(f"Target {target_name} not found")

    def update_observation_times(self, target_name, req=None, obs=None):
        """Function to update the observation times for a target."""
        for t in self.targets:
            if t["target"] == target_name:
                if req is not None:
                    t[self._req_key()] = float(req)
                if obs is not None:
                    t[self._obs_key()] = float(obs)
                t[self._rem_key()] = max(0.0, t[self._req_key()] - t[self._obs_key()])
                return
        raise ValueError(f"Target {target_name} not found")

    def save(self):
        """Function to save the prioritization file."""
        fieldnames = [
            "rank",
            "target",
            "priority",
            self._req_key(),
            self._obs_key(),
            self._rem_key(),
        ]
        with open(self.priority_file, "w", newline="") as f:
            f.write(self._generate_header())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.targets:
                writer.writerow(row)

    # refresh func to recheck all the files and make/insert new targets

    # update func to update the priorities, time req, time obs, etc. for one or more entries

    # make_priorites func to read in all files and make priorities from scratch

    # save func to save priorities file


def make_priorities(dirs=[], author="system"):
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
