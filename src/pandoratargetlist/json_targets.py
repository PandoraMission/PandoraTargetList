# Functions to make target definition JSON files

import os
import json

import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u

import exoscraper as xos
import pandorasim as psim
import pandorasat as psat
import pandorapsf as ppsf

from pandoratargetlist import __version__, HOMEDIR, TARGDEFDIR


def make_json_file(
    targets, author="System", transits=10, category="auxiliary", verbose=True
):
    """Top-level function responsible for making the JSON file for a target or
    list of targets.
    """
    target_list, aux_info, pl_flags = process_targets(targets)

    if verbose:
        print("Gathering PSFs...", end="\r")
    vda_psf = ppsf.PSF.from_name("VISDA")
    nirda_psf = ppsf.PSF.from_name("NIRDA")
    nirda_psf = nirda_psf.freeze_dimension(row=0 * u.pixel, column=0 * u.pixel)

    for i, target in enumerate(target_list):
        if verbose:
            print(
                "Running "
                + str(target)
                + " ("
                + str(i + 1)
                + "/"
                + str(len(target_list))
                + ")"
            )
        out_dict = {
            "Time Created": Time.now().value.strftime("%Y-%m-%d %H:%M:%S"),
            "Version": __version__,
            "Author": str(author),
        }

        if pl_flags[i] > 0:
            if aux_info is not None and "transits" in aux_info.columns:
                transits = aux_info.transits[i]
            out_dict.update(
                {
                    "Planet Name": target,
                    "Star Name": target[:-1],
                    "Number of Transits to Capture": transits,
                }
            )
        else:
            out_dict.update({"Star Name": target})

        if verbose:
            print("Fetching system parameters...", end="\r")
        sys_dict, info = fetch_system_dict(target, bool(pl_flags[i]), info_out=True)
        out_dict.update(sys_dict)

        if verbose:
            print("Determining best instrument parameters...", end="\r")
        instrument_dict = choose_readout_scheme(
            info, vda_psf=vda_psf, nirda_psf=nirda_psf
        )
        out_dict.update(instrument_dict)

        if verbose:
            print("Saving JSON file...", end="\r")
        # Save dictionary as JSON file
        json_file_path = (
            TARGDEFDIR
            + str(category)
            + "/"
            + str(target.replace(" ", "_"))
            + "_target_definition.json"
        )
        with open(json_file_path, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)

        print("Saved JSON file for " + str(target))


def fetch_system_dict(target, pl_flag=True, info_out=True):
    """Function to fetch information about a target and return relevant
    keywords. This will help filter between stars with and without planets.
    """
    if pl_flag:
        query_name = target[:-1]
    else:
        query_name = target

    info = xos.System.from_gaia(
        query_name, time=Time("2457389.0", format="jd", scale="tcb")
    )

    out_dict = {
        "RA": info.sky_cat["coords"].ra[0].value,
        "DEC": info.sky_cat["coords"].dec[0].value,
        "coord_epoch": "J2016.0",
        "pm_RA": info.sky_cat["coords"].pm_ra_cosdec[0].value,
        "pm_DEC": info.sky_cat["coords"].pm_dec[0].value,
        "Jmag": float(info.sky_cat["jmag"][0]),
        "Gmag": float(info.sky_cat["gmag"][0]),
        "Teff (K)": float(info.sky_cat["teff"][0].value),
        "logg": float(info.sky_cat["logg"][0]),
    }

    if pl_flag:
        targ_ind = [info[0][n].name for n in range(len(info[0].planets))].index(target)
        planet = info[0][targ_ind]

        out_dict.update(
            {
                "Planet Letter": target[-1:],
                "Period (days)": planet.pl_orbper.value,
                "Period Uncertainty (days)": planet.pl_orbper.err.value,
                "Transit Duration (hrs)": planet.pl_trandur.value,
                "Transit Epoch (BJD_TDB)": planet.pl_tranmid.value,
                "Transit Epoch Uncertainty (days)": planet.pl_tranmid.err.value,
            }
        )

        if len(info[0][:]) > 1:
            other_planets = []

            for i, pl in enumerate(info[0][:]):
                if i == targ_ind:
                    continue

                tmp_dict = {
                    "Planet Letter": str(pl)[-1],
                    "Period (days)": info[0][i].pl_orbper.value,
                    "Period Uncertainty (days)": info[0][i].pl_orbper.err.value,
                    "Transit Duration (hrs)": info[0][i].pl_trandur.value,
                    "Transit Epoch (BJD_TDB)": info[0][i].pl_tranmid.value,
                    "Transit Epoch Uncertainty (days)": info[0][i].pl_tranmid.err.value,
                }
                other_planets.append(tmp_dict)

            out_dict.update({"Additional Planets": other_planets})

    if info_out:
        return out_dict, info
    else:
        return out_dict


def choose_readout_scheme(
    info=None,
    teff=None,
    vmag=None,
    jmag=None,
    gmag=None,
    bmag=None,
    logg=4.5,
    vda_psf=None,
    nirda_psf=None,
):
    """Function to determine the brightness in NIRDA of a target and choose a
    readout scheme.
    """
    if all(x is None for x in [info, teff, vmag, jmag, gmag, bmag]):
        raise ValueError("Stellar information must be provided!")

    if info is not None:
        teff = info.sky_cat["teff"][0].value
        jmag = info.sky_cat["jmag"][0]
        # vmag = info[0].sy_vmag.value
        gmag = info.sky_cat["gmag"][0]
        bmag = info.sky_cat["bmag"][0]
        logg = info.sky_cat["logg"][0]

    # VDA
    if vda_psf is None:
        vda_psf = ppsf.PSF.from_name("VISDA")

    with open(TARGDEFDIR + "vda_readout_schemes.json", "r") as file:
        vda_schemes = json.load(file)
    vda_keys = vda_schemes["data"]["IncludedMnemonics"]

    VDA = psat.VisibleDetector()

    wav = np.arange(100, 1000) * u.nm
    s = np.trapz(VDA.sensitivity(wav), wav)
    f = VDA.flux_from_mag(bmag)
    counts = (f * s).to(u.electron / u.second).value

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

        src_flux = ((counts * u.electron / u.second) * integration_time).value.astype(
            int
        )
        data = roiscene.model(np.array([src_flux]))
        data += VDA.background_rate.value

        max_pix = np.max(data[0][0])

        if max_pix < saturation_counts:
            instrument_set = key
        else:
            break

    out_dict = {"VDA Setting": instrument_set}

    # NIRDA
    if nirda_psf is None:
        nirda_psf = ppsf.PSF.from_name("NIRDA")
        nirda_psf = nirda_psf.freeze_dimension(row=0 * u.pixel, column=0 * u.pixel)
    with open(TARGDEFDIR + "nirda_readout_schemes.json", "r") as file:
        nirda_schemes = json.load(file)
    nirda_keys = nirda_schemes["data"]["IncludedMnemonics"]

    NIRDA = psat.NIRDetector()
    integration_time = NIRDA.frame_time()

    NIRDA_trace = ppsf.TraceScene(
        np.array([[300, 40]]),
        psf=nirda_psf,
        shape=NIRDA.subarray_size,
        corner=(0, 0),
        wav_bin=1,
    )

    spectra = np.zeros((1, nirda_psf.trace_wavelength.shape[0]))
    wav, spec = psat.phoenix.get_phoenix_model(teff=teff, logg=logg, jmag=jmag)
    spectra[0, :] = nirda_psf.integrate_spectrum(wav, spec)
    spectra = spectra * u.electron / u.s

    saturation_counts = 80000
    max_pix = 0
    instrument_set = nirda_keys[0]
    for key in nirda_keys:
        nreads = nirda_schemes["data"][key]["FramesPerIntegration"]

        integration_info = psim.utils.get_integrations(
            SC_Resets1=nirda_schemes["data"][key]["SC_Resets1"],
            SC_Resets2=nirda_schemes["data"][key]["SC_Resets2"],
            SC_DropFrames1=nirda_schemes["data"][key]["SC_DropFrames1"],
            SC_DropFrames2=nirda_schemes["data"][key]["SC_DropFrames2"],
            SC_DropFrames3=nirda_schemes["data"][key]["SC_DropFrames3"],
            SC_ReadFrames=nirda_schemes["data"][key]["SC_ReadFrames"],
            SC_Groups=nirda_schemes["data"][key]["SC_Groups"],
            SC_Integrations=1,
        )
        integration_arrays = [np.hstack(idx) for idx in integration_info]
        resets = np.hstack(integration_arrays) != 1

        source_flux = (
            spectra.T[:, :, None]
            * np.ones(nreads)[None, None, :]
            * integration_time
            * resets.astype(float)
        )

        data = NIRDA_trace.model(source_flux)
        data += 8

        max_pix = np.max(np.cumsum(data, axis=0)[-1])

        if max_pix < saturation_counts:
            instrument_set = key
        else:
            break

    out_dict.update({"NIRDA Setting": instrument_set})

    return out_dict


def process_targets(input_targets, delimiter=","):
    """Function to process input target(s) for make_json_file."""
    if type(input_targets) is not str and type(input_targets) is not list:
        raise ValueError("Please make sure target input is a string!")

    aux_info = None

    if input_targets[-4] == ".":
        aux_info = pd.read_csv(input_targets, delimiter=delimiter)
        targets = aux_info["designation"].tolist()
    elif type(input_targets) is list:
        targets = [str(i) for i in input_targets]
    else:
        targets = [input_targets]

    pl_flags = np.zeros(len(targets))
    for i in range(len(targets)):
        if targets[i][:3] == "TOI":
            pl_flags[i] += 1
        else:
            try:
                int(targets[i][-1])
            except:
                pl_flags[i] += 1

    return targets, aux_info, pl_flags
