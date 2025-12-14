# A collection of useful utility functions for the target definition file generation/maintenance

from functools import lru_cache
import math

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


def query_params(name, category, return_query=False, **kwargs):
    """Function to query system parameters and output a formatted dictionary."""
    name = name.replace("_", " ")
    name = name.replace("DR3", "Gaia DR3")

    if "exoplanet" in category:
        star_name = name[:-1]
    else:
        star_name = name

    res = xos.System.from_gaia(
        star_name, time=Time("2457389.0", format="jd", scale="tcb"), **kwargs
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
        targ_ind = [res[0][n].name for n in range(len(res[0].planets))].index(
            name
        )
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
                    "Transit Epoch Uncertainty (days)": res[0][
                        i
                    ].pl_tranmid.err.value,
                }
                other_planets.append(tmp_dict)

            out_dict.update({"Additional Planets": other_planets})

    if return_query:
        return out_dict, res
    else:
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


def is_nan_or_none(value):
    """Quick helper function to determine if a value is NaN or None."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def estimate_transit_duration(period, a_rs=None, r_star=None):
    """Function to estimate the transit duration of a planet."""
    if not is_nan_or_none(a_rs) and not is_nan_or_none(r_star) and a_rs > 0:
        try:
            arc_term = r_star / a_rs
            if arc_term >= 1:
                arc_term = 0.9999  # avoid domain error
            duration = (period / math.pi) * math.asin(arc_term)
            return round(duration * 24, 4)  # convert from days to hours
        except Exception:
            pass

    # Fallback rough estimate
    return round(1.5 * (period ** (1 / 3)), 4)
