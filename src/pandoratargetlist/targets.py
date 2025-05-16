# Functions to make target definition JSON files

import os
import json
import difflib
import re
from collections import OrderedDict
import warnings

import numpy as np
from astropy.time import Time
import astropy.units as u

import pandorasim as psim
import pandorasat as psat
import pandorapsf as ppsf

from pandoratargetlist import __version__, TARGDEFDIR  # , VDA_PSF, NIRDA_PSF
from .utils import (
    star_keys,
    pl_keys,
    query_params,
    check_key_for_nan,
    load_psf_model,
    estimate_transit_duration,
    is_nan_or_none,
)


class Target(object):
    # name, category
    """
    Class to create and manipulate target definition files
    """

    def __init__(
        self,
        name=None,
        category=None,
        info_dict=None,
        author="system",
        explicit=False,
    ):
        """Ensures necessary information is present and checks for existing file"""
        # Throw error if all inputs are None
        if all(arg is None for arg in (name, category, info_dict)):
            raise ValueError("At least one input must be provided.")

        self.category = category
        self.author = author
        self.dirpath = TARGDEFDIR + self.category + "/"

        # Check if name is in info_dict
        if info_dict is not None and name is None:
            name_strs = [
                "Planet Name",
                "PlanetName",
                "Star Name",
                "StarName",
                "hostname",
                "name",
                "Name",
            ]
            lower_keys = {key.lower(): key for key in info_dict}

            for s in name_strs:
                s_lower = s.lower()
                matches = [
                    orig_key
                    for low_key, orig_key in lower_keys.items()
                    if s_lower in low_key
                ]
                if matches:
                    name_key = matches[0]
                    break
            self.name = info_dict[name_key]

        # Process name to remove spaces
        name = name.strip()
        if name[-2] == " ":
            name = name[:-2] + name[-1]
        self.name = name.replace(" ", "_")

        if not explicit:
            # Check to see if file already exists and adjust name to match
            best_match = self._crossmatch_names()

            if len(best_match) > 0:
                self.name = best_match

        self.filepath = self.dirpath + self.name + "_target_definition.json"

        # Loading in info from best match if it exists
        if info_dict is None and len(best_match) > 0:
            print("Existing file found! Loading info for " + self.name)

            with open(self.filepath, "r") as f:
                self.info = json.load(f)

        elif info_dict is not None:
            self.info = info_dict

        else:
            # Add in some basic keywords here? (Star Name, Author, etc.)
            self.info = {}

        # Checking input info against keyword structure
        self.info = self._process_keywords(self.info)

        # method to just print individual params
        # maybe add explicit=False flag to overwrite the nearest match name (in case a planet
        # file needs to exist in the same directory as a star file)

        return

    def __repr__(self):
        return f"{self.name}, {self.category} target"

    @staticmethod
    def from_name(name, category, **kwargs):
        return Target(name=name, category=category, **kwargs)

    @staticmethod
    def from_dict(info_dict, category, **kwargs):
        return Target(category=category, info_dict=info_dict)

    def _crossmatch_names(self):
        """Function to crossmatch input name with existing files"""
        filelist = [f for f in os.listdir(self.dirpath) if f[-5:] == ".json"]
        targs = [targ.split("_target")[0] for targ in filelist]

        # Looking for nearest name matches
        input_nums = re.findall(r"\d+", self.name)
        input_text = re.sub(r"\d+", "", self.name).strip()

        if "exoplanet" in self.category:
            input_end_match = re.search(r"([A-Za-z])$", self.name)
            input_suffix = (
                input_end_match.group(1) if input_end_match else None
            )

        matches = []
        for target in targs:
            targ_nums = re.findall(r"\d+", target)
            targ_text = re.sub(r"\d+", "", target).strip()

            if "exoplanet" in self.category:
                targ_end_match = re.search(r"([A-Za-z])$", target)
                targ_suffix = (
                    targ_end_match.group(1) if targ_end_match else None
                )

                if input_suffix != targ_suffix:
                    continue

            if input_nums == targ_nums:
                ratio = difflib.SequenceMatcher(
                    None, input_text.lower(), targ_text.lower()
                ).ratio()
                if ratio >= 0.6:
                    matches.append((target, ratio))

        best_matches = sorted(matches, key=lambda x: x[1], reverse=True)

        if len(best_matches) > 0:
            return best_matches[0][0]
        else:
            return []

    def _process_keywords(self, info):
        """Function to process the format and keywords of info dictionary."""
        keys = (
            [
                "Time Created",
                "Version",
                "Author",
                "Time Updated",
                "Star Name",
                "Primary Target",
            ]
            + star_keys
            + [
                "VDA Setting",
                "NIRDA Setting",
                "StarRoiDetMethod",
                "numPredefinedStarRois",
            ]
        )
        addl_planet_keys = ["Additional Planets"]
        roi_sel_keys = ["ROI_coord_epoch", "ROI_coord"]

        if "exoplanet" in self.category:
            keys = keys + pl_keys
            if "Additional Planets" in info.keys():
                keys = keys + addl_planet_keys
        if "primary" in self.category:
            keys = keys + roi_sel_keys

        reordered_info = OrderedDict()
        for key in keys:
            reordered_info[key] = info.get(key, np.nan)

        update_dict = {}

        if "coord_epoch" in info.keys():
            update_dict.update({"Coordinate Epoch": info["coord_epoch"]})

        # Check if the Star Name is specified
        if check_key_for_nan(reordered_info, "Star Name"):
            if "exoplanet" in self.category:
                update_dict.update({"Star Name": self.name[:-1]})
            else:
                update_dict.update({"Star Name": self.name})

        # Check if Primary Target is specified
        if check_key_for_nan(reordered_info, "Primary Target"):
            if any(
                s in self.category
                for s in [
                    "auxiliary",
                    "occultation",
                    "monitoring",
                    "secondary",
                ]
            ):
                update_dict.update({"Primary Target": 0})
            else:
                update_dict.update({"Primary Target": 1})

        if len(update_dict) > 0:
            update_dict.update(
                {
                    "Version": __version__,
                    "Time Updated": Time.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            reordered_info.update(update_dict)

        return reordered_info

    def make_file(self, save=True, overwrite=False, verbose=False, **kwargs):
        """Function that wraps other methods to make the target definition file in a single command"""
        # Sort input arguments
        params_args = {k: kwargs[k] for k in ["obs_window"] if k in kwargs}
        inst_args = {k: kwargs[k] for k in ["detector"] if k in kwargs}
        save_args = {k: kwargs[k] for k in ["explicit"] if k in kwargs}

        # Fetch any system params
        self.fetch_params(overwrite=overwrite, verbose=verbose, **params_args)

        # Get the instrument settings
        self.get_instrument_settings(
            overwrite=overwrite, verbose=verbose, **inst_args
        )

        # Get the ROI information
        self.get_rois(overwrite=overwrite)

        # Save file if save=True
        if save:
            self.save(**save_args)

    def fetch_params(self, overwrite=False, obs_window=None, verbose=False):
        """Function to fetch the parameters of the system"""
        keys = star_keys
        if "exoplanet" in self.category:
            keys = star_keys + pl_keys

        for key in keys:
            nan_flag = check_key_for_nan(self.info, key)
            if nan_flag:
                break

        # Check if there are any system params that need values if overwrite is False
        if nan_flag or overwrite is True:
            # Fetch system information from Gaia DR3 using exoscraper
            out_dict, self.query_result = query_params(
                self.name, self.category, return_query=True
            )

            # Fix params that might still be NaNs from the query
            if "exoplanet" in self.category:
                self._fix_pl_duration(out_dict)

            if obs_window is None:
                if "primary" in self.category or "secondary" in self.category:
                    out_dict["Obs Window (hrs)"] = 24.0
                else:
                    out_dict["Obs Window (hrs)"] = 6.0
            else:
                out_dict["Obs Window (hrs)"] = obs_window

            if overwrite is False:
                if (
                    "Additional Planets" in out_dict.keys()
                    and "Additional Planets" not in self.info.keys()
                ):
                    self.info["Additional Planets"] = out_dict[
                        "Additional Planets"
                    ]

                # Replace NaNs in info dict with values from query
                for key in out_dict.keys():
                    if key in self.info.keys() and check_key_for_nan(
                        self.info, key
                    ):
                        self.info[key] = out_dict[key]
            else:
                self.info.update(out_dict)
        elif verbose:
            print(
                "No updates made. Try overwrite=True to check for value changes."
            )

    def get_instrument_settings(
        self, detector=["VISDA", "NIRDA"], overwrite=False, verbose=False
    ):
        """Function to determine best instrument config settings."""
        if "VISDA" in detector and (
            check_key_for_nan(self.info, "VDA Setting") or overwrite
        ):
            # Put in a check for Bmag
            if check_key_for_nan(self.info, "Bmag"):
                raise ValueError(
                    "Bmag is necessary for an accurate VDA Setting"
                )

            # Load in VDA PSF
            vda_psf = load_psf_model("VISDA")
            if verbose:
                print("Loaded VDA PSF model!")

            # Read in potential VDA readout schemes
            with open(TARGDEFDIR + "vda_readout_schemes.json", "r") as file:
                vda_schemes = json.load(file)
            vda_keys = vda_schemes["data"]["IncludedMnemonics"]

            VDA = psat.VisibleDetector()

            f = VDA.mag_to_flux(self.info["Bmag"])
            counts = (f).to(u.electron / u.second).value

            roiscene = ppsf.ROIScene(
                locations=np.array([[0, 0]]),
                psf=vda_psf,
                shape=VDA.shape,
                corner=(-VDA.shape[0] // 2, -VDA.shape[1] // 2),
                ROI_size=(50, 50),
                ROI_corners=[(999, 999)],
                nROIs=1,
            )

            saturation_counts = 45000
            max_pix = 0
            instrument_set = vda_keys[0]
            for key in vda_keys:
                integration_time = VDA.integration_time

                src_flux = (
                    (counts * u.electron / u.second) * integration_time
                ).value.astype(int)
                data = roiscene.model(np.array([src_flux]))
                data += VDA.background_rate.value

                max_pix = np.max(data[0][0])

                if max_pix < saturation_counts:
                    instrument_set = key
                else:
                    break

            self.info.update({"VDA Setting": instrument_set})

        if "NIRDA" in detector and (
            check_key_for_nan(self.info, "NIRDA Setting") or overwrite
        ):
            # Put in a check for Jmag and Teff
            if check_key_for_nan(self.info, "Jmag") or check_key_for_nan(
                self.info, "Teff (K)"
            ):
                raise ValueError(
                    "Jmag and Teff are necessary for an accurate NIRDA Setting"
                )
            if check_key_for_nan(self.info, "logg"):
                logg = 4.5
            else:
                logg = self.info["logg"]

            # Load in NIRDA PSF model
            nirda_psf = load_psf_model("NIRDA")
            if verbose:
                print("Loaded NIRDA PSF model!")

            nirda_psf = nirda_psf.freeze_dimension(
                row=0 * u.pixel, column=0 * u.pixel
            )
            with open(TARGDEFDIR + "nirda_readout_schemes.json", "r") as file:
                nirda_schemes = json.load(file)
            nirda_keys = nirda_schemes["data"]["IncludedMnemonics"]

            NIRDA = psat.NIRDetector()
            integration_time = NIRDA.frame_time()

            k = nirda_psf.trace_sensitivity.value > (
                nirda_psf.trace_sensitivity.max().value * 1e-6
            )
            nirda_wavelengths = nirda_psf.trace_wavelength[k]

            NIRDA_trace = ppsf.TraceScene(
                np.array([[300, 40]]),
                psf=nirda_psf,
                shape=NIRDA.subarray_size,
                corner=(0, 0),
                wavelength=nirda_wavelengths,
                # wav_bin=1,
            )

            spectra = np.zeros((1, nirda_wavelengths.shape[0]))
            wav, spec = psat.phoenix.get_phoenix_model(
                teff=self.info["Teff (K)"], logg=logg, jmag=self.info["Jmag"]
            )
            spectra[0, :] = nirda_psf.integrate_spectrum(
                wav, spec, nirda_wavelengths
            )
            spectra = spectra * u.electron / u.s

            saturation_counts = 80000
            max_pix = 0
            instrument_set = nirda_keys[0]
            for key in nirda_keys:
                nreads = nirda_schemes["data"][key]["FramesPerIntegration"]

                integration_info = psim.utils.get_integrations(
                    SC_Resets1=nirda_schemes["data"][key]["SC_Resets1"],
                    SC_Resets2=nirda_schemes["data"][key]["SC_Resets2"],
                    SC_DropFrames1=nirda_schemes["data"][key][
                        "SC_DropFrames1"
                    ],
                    SC_DropFrames2=nirda_schemes["data"][key][
                        "SC_DropFrames2"
                    ],
                    SC_DropFrames3=nirda_schemes["data"][key][
                        "SC_DropFrames3"
                    ],
                    SC_ReadFrames=nirda_schemes["data"][key]["SC_ReadFrames"],
                    SC_Groups=nirda_schemes["data"][key]["SC_Groups"],
                    SC_Integrations=1,
                )
                integration_arrays = [
                    np.hstack(idx) for idx in integration_info
                ]
                resets = np.hstack(integration_arrays) != 1

                source_flux = (
                    spectra.T[:, :, None]
                    * np.ones(nreads)[None, None, :]
                    * integration_time
                    * resets.astype(float)
                )

                data = NIRDA_trace.model(source_flux)
                data += 8 * u.electron

                max_pix = np.max(np.cumsum(data, axis=0)[-1])

                if max_pix.value < saturation_counts:
                    instrument_set = key
                else:
                    break

            self.info.update({"NIRDA Setting": instrument_set})

    def get_rois(self, overwrite=False):
        """Function for fetching ROI parameters."""
        # This will likely be a work in progress
        keys = ["StarRoiDetMethod", "numPredefinedStarRois"]
        if "primary" in self.category:
            keys = keys + ["ROI_coord_epoch", "ROI_coord"]

        for key in keys[:-1]:
            nan_flag = check_key_for_nan(self.info, key)
            if nan_flag:
                break

        if "ROI_coord" in keys:
            for sublist in self.info["ROI_coord"]:
                for element in sublist:
                    if isinstance(element, (int, float)) and np.isnan(element):
                        nan_flag = True

        if nan_flag or overwrite is True:
            # Placeholders for later when ROI selection is fixed
            if "primary" in self.category:
                # Fix coords for now to something just randomly around target
                # Something about calculating ROIs and numROIs
                update_dict = {
                    "StarRoiDetMethod": 0,
                    "numPredefinedStarRois": 1,
                    "ROI_coord_epoch": "J2016.0",
                    "ROI_coord": [[]],
                }
            else:
                update_dict = {
                    "StarRoiDetMethod": 1,
                    "numPredefinedStarRois": 0,
                }

            if overwrite is False:
                for key in keys:
                    if key in self.info.keys() and check_key_for_nan(
                        self.info, key
                    ):
                        self.info[key] = update_dict[key]
            else:
                self.info.update(update_dict)

                # also need to add in ROI_coord_epoch and ROI_coord to process keys func

    def save(self, explicit=False):
        """Function to save the target definition file."""
        # Check if category's directory exists. If not, make it.
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        # Add in final keywords like author and update time, for example.
        if check_key_for_nan(self.info, "Time Created"):
            self.info.update(
                {
                    "Time Created": Time.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Author": self.author,
                }
            )

        self.info.update(
            {
                "Version": __version__,
                "Time Updated": Time.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # Check that keywords adhere to structure. Override this with explicit=True
        if not explicit:
            self.info = self._process_keywords(self.info)

        # Save file
        with open(self.filepath, "w") as outfile:
            json.dump(self.info, outfile, indent=4)

    def show(self):
        """Function to pretty print the information in a JSON-like format."""
        print(json.dumps(self.info, indent=4))

    def _fix_pl_duration(self, input_dict, force_query=False):
        """Helper function to fix any NaNs in planet params after the fetching them."""
        if force_query:
            out_dict, self.query_res = query_params(
                self.name, self.category, return_query=True
            )

        if is_nan_or_none(input_dict.get("Transit Duration (hrs)")):
            period = input_dict.get("Period (days)")
            epoch = input_dict.get("Transit Epoch (BJD_TDB)")
            a_rs = None
            st_rad = None

            if hasattr(self, "query_res"):
                if "pl_ratdor" in self.query_res.keys():
                    a_rs = self.query_res["pl_ratdor"]
                if "st_rad" in self.query_res.keys():
                    st_rad = self.query_res["st_rad"]

            if isinstance(period, (int, float)) and isinstance(
                epoch, (int, float)
            ):
                input_dict["Transit Duration (hrs)"] = (
                    estimate_transit_duration(period, a_rs, st_rad)
                )

        # Additional Planets
        if "Additional Planets" in input_dict.keys():
            for i, planet in enumerate(input_dict["Additional Planets"]):
                if is_nan_or_none(planet.get("Transit Duration (hrs)")):
                    period = planet.get("Period (days)")
                    epoch = planet.get("Transit Epoch (BJD_TDB)")
                    a_rs = None
                    st_rad = None

                    if hasattr(self, ".query_res"):
                        if (
                            "pl_ratdor"
                            in self.query_res["Additional Planet"][i].keys()
                        ):
                            a_rs = self.query_res["Additional Planet"][i][
                                "pl_ratdor"
                            ]
                        if "st_rad" in self.query_res.keys():
                            st_rad = self.query_res["st_rad"]

                    if isinstance(period, (int, float)) and isinstance(
                        epoch, (int, float)
                    ):
                        input_dict["Additional Planets"][i][
                            "Transit Duration (hrs)"
                        ] = estimate_transit_duration(period, a_rs, st_rad)

    def update(self, update_dict, verbose=True):
        """Function to update the value of a given parameter."""
        for key, value in update_dict.items():
            if key not in self.info.keys() and verbose:
                warnings.warn(
                    f"Warning: '{key}' is not already present. Adding it to the dictionary."
                )
            self.info[key] = value

    def delete(self, delete_keys):
        """Function to delete any keys in the dictionary"""
        if isinstance(delete_keys, str):
            delete_keys = [delete_keys]
        for key in delete_keys:
            del self.info[key]

    def delete_file(self, target_cat=None):
        """Function to remove or move this target from the given category"""
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            print(f"File '{self.filepath}' not found.")
        except PermissionError:
            print(f"Permission denied to delete '{self.filepath}'.")
