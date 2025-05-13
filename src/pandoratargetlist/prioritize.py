# Functions to prioritize targets and generate the priority files

import os
import csv

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
        self.manual_priority = set()
        self._load_or_initialize_priorities()

    def _load_json_targets(self):
        """Function for determing which targets are in a category."""
        filelist = [f for f in os.listdir(self.dirpath) if f[-5:] == ".json"]
        return sorted([targ.split("_target")[0] for targ in filelist])

    def _load_or_initialize_priorities(self):
        """Function that loads an existing prioritization file if it exists. Otherwise makes one."""
        # if self.priority_file.exist s():
        if os.path.isfile(self.priority_file):
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
        return "transits_req" if self.is_exoplanet else "hours_req"

    def _obs_key(self):
        return "transits_obs" if self.is_exoplanet else "hours_obs"

    def _rem_key(self):
        return "transits_rem" if self.is_exoplanet else "hours_rem"

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
            reader = csv.DictReader(
                (line for line in f if not line.startswith("#"))
            )
            for row in reader:
                priority = float(row["priority"])
                target_name = row["target"]
                if priority < 0.3 or priority > 0.9:
                    self.manual_priority.add(target_name)
                self.targets.append(
                    {
                        "target": target_name,
                        "rank": int(row["rank"]),
                        "priority": priority,
                        self._req_key(): float(row[self._req_key()]),
                        self._obs_key(): float(row[self._obs_key()]),
                        self._rem_key(): float(row[self._rem_key()]),
                    }
                )

    def _sync_targets(self):
        """Function to add/remove targets based on whether they're in the category."""
        existing = {t["target"]: t for t in self.targets}
        updated = []

        for name in self.json_targets:
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

        updated.sort(
            key=lambda x: x["rank"] if x["rank"] >= 0 else float("inf")
        )
        for i, t in enumerate(updated):
            t["rank"] = i
        self.targets = updated

    def _recalculate_priorities(self):
        """Function to generically recalculate priorities."""
        if self.category == "occultation-standard":
            self.targets.sort(key=lambda x: ("DR3" in x["target"], x["rank"]))

        auto_targets = [
            t for t in self.targets if t["target"] not in self.manual_priority
        ]
        manual_targets = [
            t for t in self.targets if t["target"] in self.manual_priority
        ]

        combined = manual_targets + auto_targets
        combined.sort(
            key=lambda x: (x["priority"], x["target"] in self.manual_priority),
            reverse=True,
        )

        manual_priorities = set(t["priority"] for t in manual_targets)

        n_auto = len(auto_targets)
        auto_index = 0
        for i, t in enumerate(combined):
            t["rank"] = i
            if t["target"] not in self.manual_priority:
                priority = round(
                    0.9 - (0.6 * auto_index / max(n_auto - 1, 1)), 6
                )
                # Avoid conflict with manual priorities
                while priority in manual_priorities:
                    priority = round(priority - 0.0001, 4)
                t["priority"] = priority
                auto_index += 1

        self.targets = combined

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
            (
                i
                for i, t in enumerate(self.targets)
                if t["target"] == target_name
            ),
            None,
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
                next(
                    i
                    for i, t in enumerate(self.targets)
                    if t["target"] == below
                )
                + 1
            )
        else:
            raise ValueError("Must specify new_rank, delta, above, or below")

        self.targets.insert(new_rank, target)

        # Re-rank all targets
        for i, t in enumerate(self.targets):
            t["rank"] = i

        # Update priority to be halfway between neighbors
        if 0 < new_rank < len(self.targets) - 1:
            above_priority = self.targets[new_rank - 1]["priority"]
            below_priority = self.targets[new_rank + 1]["priority"]
            target["priority"] = round(
                (above_priority + below_priority) / 2, 6
            )
        elif new_rank == 0 and len(self.targets) > 1:
            below_priority = self.targets[1]["priority"]
            target["priority"] = round(below_priority + 0.01, 6)
        elif new_rank == len(self.targets) - 1 and len(self.targets) > 1:
            above_priority = self.targets[-2]["priority"]
            target["priority"] = round(above_priority - 0.01, 6)

        self._recalculate_priorities()

    def set_priority(self, target_name, priority, lock=True):
        """Function to manually set the priority of a target."""
        for t in self.targets:
            if t["target"] == target_name:
                t["priority"] = float(priority)
                if lock:
                    self.manual_priority.add(target_name)
                else:
                    self.manual_priority.discard(target_name)
                break
        else:
            raise ValueError(f"Target {target_name} not found")
        self._recalculate_priorities()

    def clear_manual_priorities(self):
        """Function to clear any manual priorities that may have been set."""
        self.manual_priority.clear()
        self._recalculate_priorities()

    def update_observation_times(self, target_name, req=None, obs=None):
        """Function to update the observation times for a target."""
        for t in self.targets:
            if t["target"] == target_name:
                if req is not None:
                    t[self._req_key()] = float(req)
                if obs is not None:
                    t[self._obs_key()] = float(obs)
                t[self._rem_key()] = max(
                    0.0, t[self._req_key()] - t[self._obs_key()]
                )
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

    def show(self):
        """Function to pretty print the priority list."""
        key_req = self._req_key()
        key_obs = self._obs_key()
        key_rem = self._rem_key()

        print(
            "{:<5} {:<25} {:<8} {:<12} {:<12} {:<12}".format(
                "Rank", "Target", "Priority", key_req, key_obs, key_rem
            )
        )
        print("-" * 80)
        for t in self.targets:
            print(
                "{:<5} {:<25} {:<8} {:<12} {:<12} {:<12}".format(
                    t["rank"],
                    t["target"],
                    t["priority"],
                    t[key_req],
                    t[key_obs],
                    t[key_rem],
                )
            )
