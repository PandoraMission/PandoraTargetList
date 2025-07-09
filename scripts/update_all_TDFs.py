# Script to update all of the target definition files and prioritization files

from pathlib import Path

from pandoratargetlist import TARGDEFDIR
from pandoratargetlist.targets import Target
from pandoratargetlist.prioritize import Priorities

subdirs = [
    "primary-exoplanet",
    "auxiliary-exoplanet",
    "exoplanet",
    "auxiliary-standard",
    "occultation-standard",
    "monitoring-standard",
    "secondary-exoplanet",
]
verbose = True

tdf_path = Path(TARGDEFDIR)

for subdir_name in subdirs:
    subdir = tdf_path / subdir_name
    if not subdir.is_dir():
        if verbose:
            print(f"Skipping non-directory: {subdir}")
        continue

    if subdir.name != "exoplanet":
        for file in subdir.glob("*_target_definition.json"):
            target_name = file.name.replace("_target_definition.json", "")
            print(f"Running target {target_name}")
            Target.from_name(
                target_name, subdir.name, author="Ben Hord"
            ).make_file()

    print(f"Running prioritization on {subdir.name}")
    priorities = Priorities(subdir.name, author="Ben Hord")
    priorities.save()
