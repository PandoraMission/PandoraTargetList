# A collection of useful utility functions for the target definition file generation/maintenance

from functools import lru_cache

import numpy as np
from astropy.time import Time

import exoscraper as xos
import pandorapsf as ppsf


star_keys = [
    "RA",
    "DEC",
    "Coordinate Epoch",
    "pm_RA",
    "pm_DEC",
    "Jmag",
    "Gmag",
    "Bmag",
    "Teff (K)",
    "logg",
]

pl_keys = [
    "Planet Name",
    "Planet Letter",
    "Period (days)",
    "Period Uncertainty (days)",
    "Transit Duration (hrs)",
    "Transit Epoch (BJD_TDB)",
    "Transit Epoch Uncertainty (days)",
    "Obs Window (hrs)",
]


def query_params(name, category):
    """Function to query system parameters and output a formatted dictionary."""
    if "exoplanet" in category:
        star_name = name[:-1]
    else:
        star_name = name

    res = xos.System.from_gaia(
        star_name, time=Time("2457389.0", format="jd", scale="tcb")
    )

    # Make output dictionary with desired system values
    out_dict = {
        "RA": res.sky_cat["coords"].ra[0].value,
        "DEC": res.sky_cat["coords"].dec[0].value,
        "Coordinate Epoch": "J2016.0",
        "pm_RA": res.sky_cat["coords"].pm_ra_cosdec[0].value,
        "pm_DEC": res.sky_cat["coords"].pm_dec[0].value,
        "Jmag": float(res.sky_cat["jmag"][0]),
        "Gmag": float(res.sky_cat["gmag"][0]),
        "Bmag": float(res.sky_cat["bmag"][0]),
        "Teff (K)": float(res.sky_cat["teff"][0].value),
        "logg": float(res.sky_cat["logg"][0]),
    }

    # Fetch planet parameters if target is a planetary system
    if "exoplanet" in category:
        targ_ind = [res[0][n].name for n in range(len(res[0].planets))].index(name)
        planet = res[0][targ_ind]

        # Update output dictionary with planet parameters
        out_dict.update(
            {
                "Planet Name": name,
                "Planet Letter": name[-1:],
                "Period (days)": planet.pl_orbper.value,
                "Period Uncertainty (days)": planet.pl_orbper.err.value,
                "Transit Duration (hrs)": planet.pl_trandur.value,
                "Transit Epoch (BJD_TDB)": planet.pl_tranmid.value,
                "Transit Epoch Uncertainty (days)": planet.pl_tranmid.err.value,
            }
        )

        # Obtain parameters for any other planets in the system
        if len(res[0][:]) > 1:
            other_planets = []

            for i, pl in enumerate(res[0][:]):
                if i == targ_ind:
                    continue

                tmp_dict = {
                    "Planet Letter": str(pl)[-1],
                    "Period (days)": res[0][i].pl_orbper.value,
                    "Period Uncertainty (days)": res[0][i].pl_orbper.err.value,
                    "Transit Duration (hrs)": res[0][i].pl_trandur.value,
                    "Transit Epoch (BJD_TDB)": res[0][i].pl_tranmid.value,
                    "Transit Epoch Uncertainty (days)": res[0][i].pl_tranmid.err.value,
                }
                other_planets.append(tmp_dict)

            out_dict.update({"Additional Planets": other_planets})

    return out_dict


def check_key_for_nan(input_dict, key):
    value = input_dict.get(key)
    if value is None:
        return True
    try:
        if np.isnan(value):
            return True
    except TypeError:
        return False
    return False


@lru_cache()
def load_psf_model(detector: str):
    return ppsf.PSF.from_name(detector)
