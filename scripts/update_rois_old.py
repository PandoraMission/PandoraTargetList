#!/usr/bin/env python3
"""
Pandora ROI Coordinate Updater

This script updates the ROI_coord field in Pandora target definition JSON files
by querying the Gaia DR3 catalog for bright, uncrowded stars in the field of view.

Author: Generated for NASA Pandora Mission
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astroquery.gaia import Gaia
import warnings

# Constants
PLATE_SCALE = 0.789  # arcsec/pixel
DETECTOR_SIZE = 1280  # pixels
TARGET_OFFSET = 0 # 260  # arcsec
ROI_SIZE = 50  # pixels
ROI_HALF_DIAGONAL = ROI_SIZE * PLATE_SCALE * np.sqrt(2) / 2  # arcsec
QUERY_RADIUS = (
    (DETECTOR_SIZE * PLATE_SCALE / 2) - TARGET_OFFSET - ROI_HALF_DIAGONAL
)  # ~217 arcsec
MIN_SEPARATION_PIXELS = np.sqrt(2) * 25  # pixels
MIN_SEPARATION_ARCSEC = MIN_SEPARATION_PIXELS * PLATE_SCALE  # arcsec
PREFERRED_SEPARATION_ARCSEC = 20.0  # arcsec (soft constraint)
BP_MAG_MIN = 6.0
BP_MAG_MAX = 16.5
GAIA_EPOCH = 2016.0  # J2016.0 for Gaia DR3
DEFAULT_TARGET_EPOCH_BJD = 2460494.5  # July 1, 2026 in BJD_TDB
DEDUP_TOLERANCE_ARCSEC = 2.0


def bjd_tdb_to_jyear(bjd_tdb: float) -> float:
    """Convert BJD_TDB to Julian year (decimal year)."""
    # BJD_TDB is JD - 2400000.5
    jd = bjd_tdb + 2400000.5
    t = Time(jd, format="jd", scale="tdb")
    return t.jyear


def jyear_to_bjd_tdb(jyear: float) -> float:
    """Convert Julian year to BJD_TDB."""
    t = Time(jyear, format="jyear", scale="tdb")
    return t.jd - 2400000.5


def apply_proper_motion(
    ra: float,
    dec: float,
    pmra: float,
    pmdec: float,
    epoch_start: float,
    epoch_end: float,
) -> Tuple[float, float]:
    """
    Apply proper motion correction to coordinates.

    Parameters:
    -----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    pmra : float
        Proper motion in RA (mas/yr), includes cos(dec) factor
    pmdec : float
        Proper motion in Dec (mas/yr)
    epoch_start : float
        Starting epoch in Julian years
    epoch_end : float
        Target epoch in Julian years

    Returns:
    --------
    Tuple[float, float]
        Corrected (RA, Dec) in degrees
    """
    dt = epoch_end - epoch_start  # years

    # Convert proper motions from mas/yr to degrees/yr
    pmra_deg = pmra / (3600.0 * 1000.0)  # pmra already includes cos(dec)
    pmdec_deg = pmdec / (3600.0 * 1000.0)

    # Apply proper motion
    ra_new = ra + pmra_deg * dt
    dec_new = dec + pmdec_deg * dt

    return ra_new % 360, dec_new


def query_gaia_stars(
    ra: float,
    dec: float,
    radius_arcsec: float,
    bp_mag_min: float = BP_MAG_MIN,
    bp_mag_max: float = BP_MAG_MAX,
) -> pd.DataFrame:
    """
    Query Gaia DR3 for stars in a circular region.

    Parameters:
    -----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    radius_arcsec : float
        Search radius in arcseconds
    bp_mag_min : float
        Minimum Bp magnitude (brighter limit)
    bp_mag_max : float
        Maximum Bp magnitude (fainter limit)

    Returns:
    --------
    pd.DataFrame
        DataFrame with Gaia stars
    """
    coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    radius = radius_arcsec * u.arcsec

    query = f"""
    SELECT source_id, ra, dec, pmra, pmdec, phot_bp_mean_mag, phot_g_mean_mag, teff_gspphot
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600.0})
    ) = 1
    AND phot_bp_mean_mag IS NOT NULL
    AND phot_bp_mean_mag > {bp_mag_min}
    AND phot_bp_mean_mag < {bp_mag_max}
    AND pmra IS NOT NULL
    AND pmdec IS NOT NULL
    ORDER BY phot_bp_mean_mag ASC
    """

    try:
        job = Gaia.launch_job(query)
        results = job.get_results()
        return results.to_pandas()
    except Exception as e:
        warnings.warn(f"Gaia query failed: {e}")
        return pd.DataFrame()


def calculate_separation(
    ra1: float, dec1: float, ra2: float, dec2: float
) -> float:
    """
    Calculate angular separation between two coordinates in arcseconds.

    Parameters:
    -----------
    ra1, dec1 : float
        First coordinate in degrees
    ra2, dec2 : float
        Second coordinate in degrees

    Returns:
    --------
    float
        Separation in arcseconds
    """
    coord1 = SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree, frame="icrs")
    coord2 = SkyCoord(ra=ra2 * u.degree, dec=dec2 * u.degree, frame="icrs")
    return coord1.separation(coord2).arcsec


def select_roi_stars(
    target_ra: float,
    target_dec: float,
    gaia_stars: pd.DataFrame,
    json_targets: List[Dict] = None,
    target_epoch: float = DEFAULT_TARGET_EPOCH_BJD,
    max_stars: int = 8,
) -> Tuple[List[Tuple[float, float]], pd.DataFrame]:
    """
    Select stars for ROI coordinates based on brightness and crowding constraints.

    Parameters:
    -----------
    target_ra : float
        Target RA in degrees (already corrected to target epoch)
    target_dec : float
        Target Dec in degrees (already corrected to target epoch)
    gaia_stars : pd.DataFrame
        DataFrame of candidate Gaia stars
    json_targets : List[Dict]
        Optional list of JSON target dictionaries to prioritize
    target_epoch : float
        Target epoch in BJD_TDB
    max_stars : int
        Maximum number of additional stars to select (default 8)

    Returns:
    --------
    Tuple[List[Tuple[float, float]], pd.DataFrame]
        List of (RA, Dec) tuples and DataFrame with star information
    """
    target_epoch_jyear = bjd_tdb_to_jyear(target_epoch)

    # Start with target star
    selected_coords = [(target_ra, target_dec)]
    selected_info = []

    # Create candidate list starting with prioritized JSON targets
    candidates = []

    if json_targets:
        for json_target in json_targets:
            # Apply proper motion to JSON target
            json_ra, json_dec = apply_proper_motion(
                json_target["ra"],
                json_target["dec"],
                json_target["pmra"],
                json_target["pmdec"],
                GAIA_EPOCH,
                target_epoch_jyear,
            )

            # Check if within query radius
            sep = calculate_separation(
                target_ra, target_dec, json_ra, json_dec
            )
            if (
                sep <= QUERY_RADIUS and sep > 0.1
            ):  # Exclude if too close (likely the target itself)
                candidates.append(
                    {
                        "ra": json_ra,
                        "dec": json_dec,
                        "ra_orig": json_target["ra"],
                        "dec_orig": json_target["dec"],
                        "pmra": json_target["pmra"],
                        "pmdec": json_target["pmdec"],
                        "bp_mag": json_target.get("Bmag", np.nan),
                        "g_mag": json_target.get("Gmag", np.nan),
                        "teff": json_target.get("Teff (K)", np.nan),
                        "source_id": json_target.get(
                            "Star Name", "JSON_target"
                        ),
                        "distance": sep,
                        "is_json_target": True,
                    }
                )

    # Add Gaia stars
    if not gaia_stars.empty:
        for _, star in gaia_stars.iterrows():
            # Apply proper motion
            star_ra, star_dec = apply_proper_motion(
                star["ra"],
                star["dec"],
                star["pmra"],
                star["pmdec"],
                GAIA_EPOCH,
                target_epoch_jyear,
            )

            # Check if within query radius
            sep = calculate_separation(
                target_ra, target_dec, star_ra, star_dec
            )
            if sep <= QUERY_RADIUS:
                candidates.append(
                    {
                        "ra": star_ra,
                        "dec": star_dec,
                        "ra_orig": star["ra"],
                        "dec_orig": star["dec"],
                        "pmra": star["pmra"],
                        "pmdec": star["pmdec"],
                        "bp_mag": star["phot_bp_mean_mag"],
                        "g_mag": star["phot_g_mean_mag"],
                        "teff": star.get("teff_gspphot", np.nan),
                        "source_id": star["source_id"],
                        "distance": sep,
                        "is_json_target": False,
                    }
                )

    # Sort candidates: JSON targets first, then by brightness
    candidates.sort(
        key=lambda x: (
            not x["is_json_target"],
            x["bp_mag"] if not np.isnan(x["bp_mag"]) else 99,
        )
    )

    # Select stars with crowding constraints
    for candidate in candidates:
        if len(selected_coords) >= max_stars + 1:  # +1 for target
            break

        cand_ra = candidate["ra"]
        cand_dec = candidate["dec"]

        # Check minimum separation constraint (hard constraint)
        min_sep = min(
            calculate_separation(cand_ra, cand_dec, sel_ra, sel_dec)
            for sel_ra, sel_dec in selected_coords
        )

        if min_sep < MIN_SEPARATION_ARCSEC:
            continue  # Violates hard constraint

        # Passed all constraints
        selected_coords.append((cand_ra, cand_dec))
        selected_info.append(candidate)

    # Create output DataFrame
    if selected_info:
        df_info = pd.DataFrame(selected_info)
    else:
        df_info = pd.DataFrame()

    return selected_coords, df_info


def load_json_file(filepath: Path) -> Optional[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to parse {filepath}: {e}")
        return None


def save_json_file(filepath: Path, data: Dict) -> bool:
    """Save data to JSON file."""
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        warnings.warn(f"Failed to save {filepath}: {e}")
        return False


def collect_json_targets(directories: List[Path]) -> List[Dict]:
    """
    Collect target information from all JSON files in directories.

    Parameters:
    -----------
    directories : List[Path]
        List of directory paths to search

    Returns:
    --------
    List[Dict]
        List of target dictionaries with deduplicated coordinates
    """
    targets = []
    seen_coords = []

    for directory in directories:
        for json_file in directory.rglob("*.json"):
            data = load_json_file(json_file)
            if data and "RA" in data and "DEC" in data:
                ra = data["RA"]
                dec = data["DEC"]

                # Check for duplicates
                is_duplicate = False
                for seen_ra, seen_dec in seen_coords:
                    if (
                        calculate_separation(ra, dec, seen_ra, seen_dec)
                        < DEDUP_TOLERANCE_ARCSEC
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    targets.append(
                        {
                            "ra": ra,
                            "dec": dec,
                            "pmra": data.get("pm_RA", 0.0),
                            "pmdec": data.get("pm_DEC", 0.0),
                            "Bmag": data.get("Bmag", np.nan),
                            "Gmag": data.get("Gmag", np.nan),
                            "Teff (K)": data.get("Teff (K)", np.nan),
                            "Star Name": data.get("Star Name", "Unknown"),
                        }
                    )
                    seen_coords.append((ra, dec))

    return targets


def process_json_files(
    directories: List[Path],
    target_epoch: float = DEFAULT_TARGET_EPOCH_BJD,
    save_updates: bool = False,
    prioritize_json: bool = False,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Process all JSON files in directories and update ROI coordinates.

    Parameters:
    -----------
    directories : List[Path]
        List of directory paths to search
    target_epoch : float
        Target epoch in BJD_TDB
    save_updates : bool
        Whether to save updates to JSON files
    prioritize_json : bool
        Whether to prioritize other JSON targets in ROI selection
    output_csv : Optional[Path]
        Path to save CSV with star information

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all selected stars
    """
    all_stars_info = []
    json_targets = None

    if prioritize_json:
        print("Collecting JSON targets for prioritization...")
        json_targets = collect_json_targets(directories)
        print(f"Found {len(json_targets)} unique JSON targets")

    target_epoch_jyear = bjd_tdb_to_jyear(target_epoch)
    files_processed = 0
    files_updated = 0

    for directory in directories:
        for json_file in directory.rglob("*.json"):
            data = load_json_file(json_file)

            if not data or "ROI_coord" not in data:
                continue

            files_processed += 1

            # Extract target information
            target_ra_orig = data["RA"]
            target_dec_orig = data["DEC"]
            target_pmra = data.get("pm_RA", 0.0)
            target_pmdec = data.get("pm_DEC", 0.0)

            # Apply proper motion to target
            target_ra, target_dec = apply_proper_motion(
                target_ra_orig,
                target_dec_orig,
                target_pmra,
                target_pmdec,
                GAIA_EPOCH,
                target_epoch_jyear,
            )

            print(f"\nProcessing: {json_file.name}")
            print(
                f"  Target: {data.get('Star Name', 'Unknown')} at ({target_ra:.6f}, {target_dec:.6f})"
            )

            # Query Gaia
            gaia_stars = query_gaia_stars(target_ra, target_dec, QUERY_RADIUS)
            print(f"  Found {len(gaia_stars)} Gaia stars in field")

            # Select ROI stars
            roi_coords, stars_info = select_roi_stars(
                target_ra,
                target_dec,
                gaia_stars,
                json_targets=json_targets,
                target_epoch=target_epoch,
                max_stars=8,
            )

            print(
                f"  Selected {len(roi_coords)} total ROI coordinates (including target)"
            )

            # Update JSON data
            data["ROI_coord"] = roi_coords
            data["numPredefinedStarRois"] = len(roi_coords)

            # Save if requested
            if save_updates:
                if save_json_file(json_file, data):
                    files_updated += 1
                    print(f"  Updated {json_file}")

            # Add to combined results
            if not stars_info.empty:
                stars_info["target_file"] = json_file.name
                stars_info["target_name"] = data.get("Star Name", "Unknown")
                stars_info["target_ra"] = target_ra
                stars_info["target_dec"] = target_dec
                all_stars_info.append(stars_info)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Files processed: {files_processed}")
    if save_updates:
        print(f"Files updated: {files_updated}")

    # Combine all star info
    if all_stars_info:
        combined_df = pd.concat(all_stars_info, ignore_index=True)

        # Reorder columns
        cols = [
            "target_file",
            "target_name",
            "target_ra",
            "target_dec",
            "source_id",
            "ra",
            "dec",
            "ra_orig",
            "dec_orig",
            "pmra",
            "pmdec",
            "bp_mag",
            "g_mag",
            "teff",
            "distance",
            "is_json_target",
        ]
        combined_df = combined_df[cols]

        if output_csv:
            combined_df.to_csv(output_csv, index=False)
            print(f"Saved star information to: {output_csv}")

        return combined_df
    else:
        return pd.DataFrame()


def process_coordinates(
    ra: float,
    dec: float,
    pmra: Optional[float] = None,
    pmdec: Optional[float] = None,
    epoch: Optional[float] = None,
    target_epoch: float = DEFAULT_TARGET_EPOCH_BJD,
    prioritize_json: bool = False,
    json_directories: Optional[List[Path]] = None,
) -> Tuple[List[Tuple[float, float]], pd.DataFrame]:
    """
    Find ROI coordinates for a given set of coordinates.

    Parameters:
    -----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    pmra : Optional[float]
        Proper motion in RA (mas/yr), includes cos(dec)
    pmdec : Optional[float]
        Proper motion in Dec (mas/yr)
    epoch : Optional[float]
        Coordinate epoch in BJD_TDB
    target_epoch : float
        Target epoch in BJD_TDB
    prioritize_json : bool
        Whether to prioritize JSON targets
    json_directories : Optional[List[Path]]
        Directories to search for JSON targets

    Returns:
    --------
    Tuple[List[Tuple[float, float]], pd.DataFrame]
        ROI coordinates and star information
    """
    target_epoch_jyear = bjd_tdb_to_jyear(target_epoch)

    # Apply proper motion if provided
    if pmra is not None and pmdec is not None and epoch is not None:
        epoch_jyear = bjd_tdb_to_jyear(epoch)
        target_ra, target_dec = apply_proper_motion(
            ra, dec, pmra, pmdec, epoch_jyear, target_epoch_jyear
        )
        print(
            f"Applied proper motion: ({ra:.6f}, {dec:.6f}) -> ({target_ra:.6f}, {target_dec:.6f})"
        )
    else:
        target_ra, target_dec = ra, dec
        print(
            f"Using coordinates as-is (assumed already corrected): ({ra:.6f}, {dec:.6f})"
        )

    # Collect JSON targets if requested
    json_targets = None
    if prioritize_json and json_directories:
        print("Collecting JSON targets for prioritization...")
        json_targets = collect_json_targets(json_directories)
        print(f"Found {len(json_targets)} unique JSON targets")

    # Query Gaia
    print(f"Querying Gaia DR3 within {QUERY_RADIUS:.1f} arcsec...")
    gaia_stars = query_gaia_stars(target_ra, target_dec, QUERY_RADIUS)
    print(f"Found {len(gaia_stars)} Gaia stars in field")

    # Select ROI stars
    roi_coords, stars_info = select_roi_stars(
        target_ra,
        target_dec,
        gaia_stars,
        json_targets=json_targets,
        target_epoch=target_epoch,
        max_stars=8,
    )

    print(
        f"Selected {len(roi_coords)} total ROI coordinates (including target)"
    )

    return roi_coords, stars_info


def main():
    parser = argparse.ArgumentParser(
        description="Update ROI coordinates in Pandora target definition JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all JSON files in a directory (dry run)
  python roi_updater.py --directory /path/to/targets
  
  # Process and save updates
  python roi_updater.py --directory /path/to/targets --save
  
  # Process with JSON target prioritization
  python roi_updater.py --directory /path/to/targets --prioritize-json --save
  
  # Process multiple directories with custom epoch
  python roi_updater.py -d /path/dir1 -d /path/dir2 --target-epoch 2460600.5 --save
  
  # Find ROI coords for specific coordinates
  python roi_updater.py --coords 209.3886 43.4933
  
  # Find ROI coords with proper motion correction
  python roi_updater.py --coords 209.3886 43.4933 --pm -134.79 -44.23 --epoch 2457388.5
  
  # Save results to CSV
  python roi_updater.py --directory /path/to/targets --output stars_info.csv
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--directory",
        "-d",
        action="append",
        type=Path,
        help="Directory to search for JSON files (can specify multiple times)",
    )
    input_group.add_argument(
        "--coords",
        nargs=2,
        type=float,
        metavar=("RA", "DEC"),
        help="Process specific coordinates (RA, Dec in degrees)",
    )

    # Proper motion options (for --coords mode)
    parser.add_argument(
        "--pm",
        nargs=2,
        type=float,
        metavar=("PMRA", "PMDEC"),
        help="Proper motion (mas/yr) for coordinate mode",
    )
    parser.add_argument(
        "--epoch",
        type=float,
        help="Coordinate epoch in BJD_TDB (for --coords mode with --pm)",
    )

    # Processing options
    parser.add_argument(
        "--target-epoch",
        type=float,
        default=DEFAULT_TARGET_EPOCH_BJD,
        help=f"Target epoch in BJD_TDB (default: {DEFAULT_TARGET_EPOCH_BJD} = July 1, 2026)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save updates to JSON files (default: dry run)",
    )
    parser.add_argument(
        "--prioritize-json",
        action="store_true",
        help="Prioritize other JSON targets in ROI selection",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output CSV file for star information",
    )

    args = parser.parse_args()

    # Validate proper motion arguments
    if args.pm and not args.epoch:
        parser.error("--pm requires --epoch to be specified")
    if args.epoch and not args.pm:
        parser.error("--epoch requires --pm to be specified")

    # Process based on mode
    if args.coords:
        # Coordinate mode
        ra, dec = args.coords
        pmra = args.pm[0] if args.pm else None
        pmdec = args.pm[1] if args.pm else None
        epoch = args.epoch

        json_dirs = None
        if args.prioritize_json:
            if not args.directory:
                parser.error(
                    "--prioritize-json requires --directory to be specified"
                )
            json_dirs = args.directory

        roi_coords, stars_info = process_coordinates(
            ra,
            dec,
            pmra,
            pmdec,
            epoch,
            target_epoch=args.target_epoch,
            prioritize_json=args.prioritize_json,
            json_directories=json_dirs,
        )

        print("\nROI Coordinates:")
        for i, (roi_ra, roi_dec) in enumerate(roi_coords):
            print(f"  {i}: ({roi_ra:.6f}, {roi_dec:.6f})")

        if not stars_info.empty:
            print("\nSelected Stars:")
            print(
                stars_info[
                    [
                        "source_id",
                        "ra",
                        "dec",
                        "bp_mag",
                        "distance",
                        "is_json_target",
                    ]
                ].to_string()
            )

            if args.output:
                stars_info.to_csv(args.output, index=False)
                print(f"\nSaved star information to: {args.output}")

    else:
        # Directory mode
        if not args.directory:
            parser.error("--directory is required when not using --coords")

        output_csv = args.output if args.output else Path("roi_stars_info.csv")

        combined_df = process_json_files(
            args.directory,
            target_epoch=args.target_epoch,
            save_updates=args.save,
            prioritize_json=args.prioritize_json,
            output_csv=output_csv,
        )

        if not combined_df.empty:
            print(
                f"\nTotal stars selected across all targets: {len(combined_df)}"
            )


if __name__ == "__main__":
    main()
