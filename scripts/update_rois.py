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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

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
    t = Time(jd, format='jd', scale='tdb')
    return t.jyear


def jyear_to_bjd_tdb(jyear: float) -> float:
    """Convert Julian year to BJD_TDB."""
    t = Time(jyear, format='jyear', scale='tdb')
    return t.jd - 2400000.5


def apply_proper_motion(ra: float, dec: float, pmra: float, pmdec: float,
                       epoch_start: float, epoch_end: float) -> Tuple[float, float]:
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
    
    return ra_new, dec_new


def query_gaia_stars(ra: float, dec: float, radius_arcsec: float,
                     bp_mag_min: float = BP_MAG_MIN,
                     bp_mag_max: float = BP_MAG_MAX) -> pd.DataFrame:
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
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
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


def calculate_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
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
    coord1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree, frame='icrs')
    coord2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree, frame='icrs')
    return coord1.separation(coord2).arcsec


def select_roi_stars(target_ra: float, target_dec: float,
                    gaia_stars: pd.DataFrame,
                    json_targets: List[Dict] = None,
                    target_epoch: float = DEFAULT_TARGET_EPOCH_BJD,
                    max_stars: int = 8) -> Tuple[List[Tuple[float, float]], pd.DataFrame]:
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
                json_target['ra'], json_target['dec'],
                json_target['pmra'], json_target['pmdec'],
                GAIA_EPOCH, target_epoch_jyear
            )
            
            # Check if within query radius
            sep = calculate_separation(target_ra, target_dec, json_ra, json_dec)
            if sep <= QUERY_RADIUS and sep > 0.1:  # Exclude if too close (likely the target itself)
                candidates.append({
                    'ra': json_ra,
                    'dec': json_dec,
                    'ra_orig': json_target['ra'],
                    'dec_orig': json_target['dec'],
                    'pmra': json_target['pmra'],
                    'pmdec': json_target['pmdec'],
                    'bp_mag': json_target.get('Bmag', np.nan),
                    'g_mag': json_target.get('Gmag', np.nan),
                    'teff': json_target.get('Teff (K)', np.nan),
                    'source_id': json_target.get('Star Name', 'JSON_target'),
                    'distance': sep,
                    'is_json_target': True
                })
    
    # Add Gaia stars
    if not gaia_stars.empty:
        for _, star in gaia_stars.iterrows():
            # Apply proper motion
            star_ra, star_dec = apply_proper_motion(
                star['ra'], star['dec'],
                star['pmra'], star['pmdec'],
                GAIA_EPOCH, target_epoch_jyear
            )
            
            # Check if within query radius
            sep = calculate_separation(target_ra, target_dec, star_ra, star_dec)
            if sep <= QUERY_RADIUS:
                candidates.append({
                    'ra': star_ra,
                    'dec': star_dec,
                    'ra_orig': star['ra'],
                    'dec_orig': star['dec'],
                    'pmra': star['pmra'],
                    'pmdec': star['pmdec'],
                    'bp_mag': star['phot_bp_mean_mag'],
                    'g_mag': star['phot_g_mean_mag'],
                    'teff': star.get('teff_gspphot', np.nan),
                    'source_id': star['source_id'],
                    'distance': sep,
                    'is_json_target': False
                })
    
    # Sort candidates: JSON targets first, then by brightness
    candidates.sort(key=lambda x: (not x['is_json_target'], x['bp_mag'] if not np.isnan(x['bp_mag']) else 99))
    
    # Select stars with crowding constraints
    for candidate in candidates:
        if len(selected_coords) >= max_stars + 1:  # +1 for target
            break
        
        cand_ra = candidate['ra']
        cand_dec = candidate['dec']
        
        # Check minimum separation constraint (hard constraint)
        min_sep = min(calculate_separation(cand_ra, cand_dec, sel_ra, sel_dec)
                     for sel_ra, sel_dec in selected_coords)
        
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
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to parse {filepath}: {e}")
        return None


def save_json_file(filepath: Path, data: Dict) -> bool:
    """Save data to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        warnings.warn(f"Failed to save {filepath}: {e}")
        return False


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file contents for deduplication."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return None


def load_json_file_with_info(filepath: Path) -> Tuple[Path, Optional[Dict], Optional[str]]:
    """
    Load JSON file and return path, data, and content hash.
    Helper function for parallel processing.
    """
    data = load_json_file(filepath)
    file_hash = compute_file_hash(filepath) if data else None
    return filepath, data, file_hash


def collect_json_targets(directories: List[Path], use_parallel: bool = True) -> List[Dict]:
    """
    Collect target information from all JSON files in directories.
    Optimized version with parallel processing and efficient deduplication.
    
    Parameters:
    -----------
    directories : List[Path]
        List of directory paths to search
    use_parallel : bool
        Whether to use parallel processing (default True)
    
    Returns:
    --------
    List[Dict]
        List of target dictionaries with deduplicated coordinates
    """
    print("Scanning directories for JSON files...")
    
    # First, collect all JSON file paths
    json_files = []
    for directory in directories:
        json_files.extend(directory.rglob("*.json"))
    
    print(f"Found {len(json_files)} JSON files, loading and deduplicating...")
    
    # Track seen files by content hash and coordinates
    seen_hashes = set()
    seen_coords = []
    targets = []
    
    # Load files in parallel if requested
    if use_parallel and len(json_files) > 10:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(load_json_file_with_info, f): f for f in json_files}
            
            for i, future in enumerate(as_completed(futures), 1):
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(json_files)} files...")
                
                try:
                    filepath, data, file_hash = future.result()
                    
                    if not data or 'RA' not in data or 'DEC' not in data:
                        continue
                    
                    # Skip if we've seen this exact file content before
                    if file_hash and file_hash in seen_hashes:
                        continue
                    
                    ra = data['RA']
                    dec = data['DEC']
                    
                    # Check for coordinate duplicates
                    is_duplicate = False
                    for seen_ra, seen_dec in seen_coords:
                        if calculate_separation(ra, dec, seen_ra, seen_dec) < DEDUP_TOLERANCE_ARCSEC:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        targets.append({
                            'ra': ra,
                            'dec': dec,
                            'pmra': data.get('pm_RA', 0.0),
                            'pmdec': data.get('pm_DEC', 0.0),
                            'Bmag': data.get('Bmag', np.nan),
                            'Gmag': data.get('Gmag', np.nan),
                            'Teff (K)': data.get('Teff (K)', np.nan),
                            'Star Name': data.get('Star Name', 'Unknown')
                        })
                        seen_coords.append((ra, dec))
                        if file_hash:
                            seen_hashes.add(file_hash)
                
                except Exception as e:
                    warnings.warn(f"Error processing file: {e}")
    else:
        # Sequential processing for small numbers of files
        for i, json_file in enumerate(json_files, 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(json_files)} files...")
            
            data = load_json_file(json_file)
            if not data or 'RA' not in data or 'DEC' not in data:
                continue
            
            file_hash = compute_file_hash(json_file)
            
            # Skip if we've seen this exact file content before
            if file_hash and file_hash in seen_hashes:
                continue
            
            ra = data['RA']
            dec = data['DEC']
            
            # Check for coordinate duplicates
            is_duplicate = False
            for seen_ra, seen_dec in seen_coords:
                if calculate_separation(ra, dec, seen_ra, seen_dec) < DEDUP_TOLERANCE_ARCSEC:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                targets.append({
                    'ra': ra,
                    'dec': dec,
                    'pmra': data.get('pm_RA', 0.0),
                    'pmdec': data.get('pm_DEC', 0.0),
                    'Bmag': data.get('Bmag', np.nan),
                    'Gmag': data.get('Gmag', np.nan),
                    'Teff (K)': data.get('Teff (K)', np.nan),
                    'Star Name': data.get('Star Name', 'Unknown')
                })
                seen_coords.append((ra, dec))
                if file_hash:
                    seen_hashes.add(file_hash)
    
    print(f"Loaded {len(targets)} unique targets (removed {len(json_files) - len(targets)} duplicates)")
    return targets


def plot_fov_diagnostic(target_ra: float, target_dec: float,
                        roi_coords: List[Tuple[float, float]],
                        stars_info: pd.DataFrame,
                        target_name: str = "Unknown",
                        output_path: Optional[Path] = None,
                        show_plot: bool = True):
    """
    Create diagnostic plot showing FOV, ROIs, and selected stars.
    
    Parameters:
    -----------
    target_ra : float
        Target RA in degrees
    target_dec : float
        Target Dec in degrees
    roi_coords : List[Tuple[float, float]]
        List of (RA, Dec) tuples for ROI stars
    stars_info : pd.DataFrame
        DataFrame with star information
    target_name : str
        Name of target for plot title
    output_path : Optional[Path]
        Path to save plot
    show_plot : bool
        Whether to display plot interactively
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Convert to relative coordinates (arcsec from target)
    def to_relative_coords(ra, dec):
        # Simple tangent plane projection for small fields
        dra = (ra - target_ra) * 3600 * np.cos(np.radians(target_dec))
        ddec = (dec - target_dec) * 3600
        return dra, ddec
    
    # Plot detector FOV (square, centered at origin with target offset)
    fov_size = DETECTOR_SIZE * PLATE_SCALE
    detector_rect = Rectangle(
        (-fov_size/2, -fov_size/2),
        fov_size, fov_size,
        fill=False, edgecolor='black', linewidth=2, label='Detector FOV'
    )
    ax.add_patch(detector_rect)
    
    # Plot target position (offset from center)
    target_x, target_y = 0, 0  # Target is our reference point
    ax.plot(target_x, target_y, 'r*', markersize=20, label='Target Star', zorder=5)
    
    # Plot query radius
    query_circle = Circle(
        (target_x, target_y), QUERY_RADIUS,
        fill=False, edgecolor='blue', linewidth=1.5, 
        linestyle='--', label=f'Query Radius ({QUERY_RADIUS:.0f}")'
    )
    ax.add_patch(query_circle)
    
    # Plot minimum separation circle
    min_sep_circle = Circle(
        (target_x, target_y), MIN_SEPARATION_ARCSEC,
        fill=False, edgecolor='orange', linewidth=1, 
        linestyle=':', label=f'Min Separation ({MIN_SEPARATION_ARCSEC:.1f}")'
    )
    ax.add_patch(min_sep_circle)
    
    # Plot all candidate stars from Gaia (if available in stars_info)
    if not stars_info.empty:
        # Plot selected ROI stars
        for i, (ra, dec) in enumerate(roi_coords[1:], 1):  # Skip target (first coord)
            x, y = to_relative_coords(ra, dec)
            
            # Determine if JSON target or Gaia star
            is_json = False
            if i <= len(stars_info):
                is_json = stars_info.iloc[i-1]['is_json_target']
            
            color = 'green' if is_json else 'blue'
            marker = 's' if is_json else 'o'
            ax.plot(x, y, marker, color=color, markersize=10, 
                   label='JSON Target' if (i == 1 and is_json) else ('Gaia Star' if i == 1 else ''),
                   zorder=4)
            
            # Plot ROI box around each selected star
            roi_size_arcsec = ROI_SIZE * PLATE_SCALE
            roi_rect = Rectangle(
                (x - roi_size_arcsec/2, y - roi_size_arcsec/2),
                roi_size_arcsec, roi_size_arcsec,
                fill=False, edgecolor=color, linewidth=1, alpha=0.5
            )
            ax.add_patch(roi_rect)
            
            # Add star label
            ax.text(x + 5, y + 5, f"{i}", fontsize=8, color=color)
    
    # Plot target ROI
    roi_size_arcsec = ROI_SIZE * PLATE_SCALE
    target_roi = Rectangle(
        (target_x - roi_size_arcsec/2, target_y - roi_size_arcsec/2),
        roi_size_arcsec, roi_size_arcsec,
        fill=False, edgecolor='red', linewidth=1.5, alpha=0.7
    )
    ax.add_patch(target_roi)
    
    # Plot possible target positions (circle at TARGET_OFFSET radius from center)
    # This shows where the target could be placed depending on roll angle
    center_x, center_y = -target_x, -target_y  # Detector center relative to target
    offset_circle = Circle(
        (center_x, center_y), TARGET_OFFSET,
        fill=False, edgecolor='red', linewidth=1, 
        linestyle='-.', label=f'Target Offset Circle ({TARGET_OFFSET}")',
        alpha=0.5
    )
    ax.add_patch(offset_circle)
    
    # Mark detector center
    ax.plot(center_x, center_y, 'kx', markersize=15, 
           markeredgewidth=2, label='Detector Center')
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('ΔRA (arcsec)', fontsize=12)
    ax.set_ylabel('ΔDec (arcsec)', fontsize=12)
    ax.set_title(f'FOV Diagnostic: {target_name}\n'
                f'RA={target_ra:.6f}°, Dec={target_dec:.6f}°\n'
                f'Selected {len(roi_coords)} ROI stars (including target)',
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Set plot limits to show full FOV plus some margin
    margin = 100  # arcsec
    plot_limit = max(fov_size/2, QUERY_RADIUS) + margin
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    
    # Add text box with statistics
    stats_text = f'ROI Size: {ROI_SIZE}×{ROI_SIZE} px ({roi_size_arcsec:.1f}")\n'
    stats_text += f'Plate Scale: {PLATE_SCALE} "/px\n'
    stats_text += f'Query Radius: {QUERY_RADIUS:.1f}"\n'
    stats_text += f'Min Separation: {MIN_SEPARATION_ARCSEC:.1f}"'
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plot to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def process_json_files(directories: List[Path],
                      target_epoch: float = DEFAULT_TARGET_EPOCH_BJD,
                      save_updates: bool = False,
                      prioritize_json: bool = False,
                      output_csv: Optional[Path] = None,
                      create_plots: bool = False,
                      plot_dir: Optional[Path] = None,
                      show_plots: bool = False) -> pd.DataFrame:
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
    create_plots : bool
        Whether to create diagnostic plots
    plot_dir : Optional[Path]
        Directory to save diagnostic plots
    show_plots : bool
        Whether to display plots interactively
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all selected stars
    """
    all_stars_info = []
    json_targets = None
    
    if prioritize_json:
        json_targets = collect_json_targets(directories)
        print(f"Found {len(json_targets)} unique JSON targets for prioritization\n")
    
    target_epoch_jyear = bjd_tdb_to_jyear(target_epoch)
    files_processed = 0
    files_updated = 0
    
    # Create plot directory if needed
    if create_plots and plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    for directory in directories:
        for json_file in directory.rglob("*.json"):
            data = load_json_file(json_file)
            
            if not data or 'ROI_coord' not in data:
                continue
            
            files_processed += 1
            
            # Extract target information
            target_ra_orig = data['RA']
            target_dec_orig = data['DEC']
            target_pmra = data.get('pm_RA', 0.0)
            target_pmdec = data.get('pm_DEC', 0.0)
            
            # Apply proper motion to target
            target_ra, target_dec = apply_proper_motion(
                target_ra_orig, target_dec_orig,
                target_pmra, target_pmdec,
                GAIA_EPOCH, target_epoch_jyear
            )
            
            target_name = data.get('Star Name', 'Unknown')
            print(f"\nProcessing: {json_file.name}")
            print(f"  Target: {target_name} at ({target_ra:.6f}, {target_dec:.6f})")
            
            # Query Gaia
            gaia_stars = query_gaia_stars(target_ra, target_dec, QUERY_RADIUS)
            print(f"  Found {len(gaia_stars)} Gaia stars in field")
            
            # Select ROI stars
            roi_coords, stars_info = select_roi_stars(
                target_ra, target_dec, gaia_stars,
                json_targets=json_targets,
                target_epoch=target_epoch,
                max_stars=8
            )
            
            print(f"  Selected {len(roi_coords)} total ROI coordinates (including target)")
            
            # Update JSON data
            data['ROI_coord'] = roi_coords
            data['numPredefinedStarRois'] = len(roi_coords)
            
            # Save if requested
            if save_updates:
                if save_json_file(json_file, data):
                    files_updated += 1
                    print(f"  Updated {json_file}")
            
            # Create diagnostic plot if requested
            if create_plots:
                plot_path = None
                if plot_dir:
                    plot_filename = f"{json_file.stem}_fov_diagnostic.png"
                    plot_path = plot_dir / plot_filename
                
                plot_fov_diagnostic(
                    target_ra, target_dec, roi_coords, stars_info,
                    target_name=target_name,
                    output_path=plot_path,
                    show_plot=show_plots
                )
            
            # Add to combined results
            if not stars_info.empty:
                stars_info['target_file'] = json_file.name
                stars_info['target_name'] = target_name
                stars_info['target_ra'] = target_ra
                stars_info['target_dec'] = target_dec
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
        cols = ['target_file', 'target_name', 'target_ra', 'target_dec',
                'source_id', 'ra', 'dec', 'ra_orig', 'dec_orig',
                'pmra', 'pmdec', 'bp_mag', 'g_mag', 'teff',
                'distance', 'is_json_target']
        combined_df = combined_df[cols]
        
        if output_csv:
            combined_df.to_csv(output_csv, index=False)
            print(f"Saved star information to: {output_csv}")
        
        return combined_df
    else:
        return pd.DataFrame()


def process_coordinates(ra: float, dec: float,
                       pmra: Optional[float] = None,
                       pmdec: Optional[float] = None,
                       epoch: Optional[float] = None,
                       target_epoch: float = DEFAULT_TARGET_EPOCH_BJD,
                       prioritize_json: bool = False,
                       json_directories: Optional[List[Path]] = None,
                       create_plot: bool = False,
                       plot_path: Optional[Path] = None,
                       show_plot: bool = False) -> Tuple[List[Tuple[float, float]], pd.DataFrame]:
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
    create_plot : bool
        Whether to create diagnostic plot
    plot_path : Optional[Path]
        Path to save plot
    show_plot : bool
        Whether to display plot
    
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
        print(f"Applied proper motion: ({ra:.6f}, {dec:.6f}) -> ({target_ra:.6f}, {target_dec:.6f})")
    else:
        target_ra, target_dec = ra, dec
        print(f"Using coordinates as-is (assumed already corrected): ({ra:.6f}, {dec:.6f})")
    
    # Collect JSON targets if requested
    json_targets = None
    if prioritize_json and json_directories:
        json_targets = collect_json_targets(json_directories)
        print(f"Found {len(json_targets)} unique JSON targets\n")
    
    # Query Gaia
    print(f"Querying Gaia DR3 within {QUERY_RADIUS:.1f} arcsec...")
    gaia_stars = query_gaia_stars(target_ra, target_dec, QUERY_RADIUS)
    print(f"Found {len(gaia_stars)} Gaia stars in field")
    
    # Select ROI stars
    roi_coords, stars_info = select_roi_stars(
        target_ra, target_dec, gaia_stars,
        json_targets=json_targets,
        target_epoch=target_epoch,
        max_stars=8
    )
    
    print(f"Selected {len(roi_coords)} total ROI coordinates (including target)")
    
    # Create diagnostic plot if requested
    if create_plot:
        plot_fov_diagnostic(
            target_ra, target_dec, roi_coords, stars_info,
            target_name=f"RA={target_ra:.4f}, Dec={target_dec:.4f}",
            output_path=plot_path,
            show_plot=show_plot
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
  
  # Process and save updates with diagnostic plots
  python roi_updater.py --directory /path/to/targets --save --plot --plot-dir ./plots
  
  # Process with JSON target prioritization
  python roi_updater.py --directory /path/to/targets --prioritize-json --save
  
  # Process multiple directories with custom epoch
  python roi_updater.py -d /path/dir1 -d /path/dir2 --target-epoch 2460600.5 --save
  
  # Find ROI coords for specific coordinates with plot
  python roi_updater.py --coords 209.3886 43.4933 --plot --show-plot
  
  # Find ROI coords with proper motion correction
  python roi_updater.py --coords 209.3886 43.4933 --pm -134.79 -44.23 --epoch 2457388.5
  
  # Save results to CSV
  python roi_updater.py --directory /path/to/targets --output stars_info.csv
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--directory', '-d', action='append', type=Path,
                           help='Directory to search for JSON files (can specify multiple times)')
    input_group.add_argument('--coords', nargs=2, type=float, metavar=('RA', 'DEC'),
                           help='Process specific coordinates (RA, Dec in degrees)')
    
    # Proper motion options (for --coords mode)
    parser.add_argument('--pm', nargs=2, type=float, metavar=('PMRA', 'PMDEC'),
                       help='Proper motion (mas/yr) for coordinate mode')
    parser.add_argument('--epoch', type=float,
                       help='Coordinate epoch in BJD_TDB (for --coords mode with --pm)')
    
    # Processing options
    parser.add_argument('--target-epoch', type=float, default=DEFAULT_TARGET_EPOCH_BJD,
                       help=f'Target epoch in BJD_TDB (default: {DEFAULT_TARGET_EPOCH_BJD} = July 1, 2026)')
    parser.add_argument('--save', action='store_true',
                       help='Save updates to JSON files (default: dry run)')
    parser.add_argument('--prioritize-json', action='store_true',
                       help='Prioritize other JSON targets in ROI selection')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output CSV file for star information')
    
    # Plotting options
    parser.add_argument('--plot', action='store_true',
                       help='Create diagnostic plots showing FOV and ROI positions')
    parser.add_argument('--plot-dir', type=Path,
                       help='Directory to save diagnostic plots (default: ./roi_plots)')
    parser.add_argument('--show-plot', action='store_true',
                       help='Display plots interactively (default: save only)')
    
    args = parser.parse_args()
    
    # Validate proper motion arguments
    if args.pm and not args.epoch:
        parser.error("--pm requires --epoch to be specified")
    if args.epoch and not args.pm:
        parser.error("--epoch requires --pm to be specified")
    
    # Set default plot directory
    if args.plot and not args.plot_dir:
        args.plot_dir = Path('./roi_plots')
    
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
                parser.error("--prioritize-json requires --directory to be specified")
            json_dirs = args.directory
        
        plot_path = None
        if args.plot and args.plot_dir:
            args.plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = args.plot_dir / f"coord_{ra:.4f}_{dec:.4f}_diagnostic.png"
        
        roi_coords, stars_info = process_coordinates(
            ra, dec, pmra, pmdec, epoch,
            target_epoch=args.target_epoch,
            prioritize_json=args.prioritize_json,
            json_directories=json_dirs,
            create_plot=args.plot,
            plot_path=plot_path,
            show_plot=args.show_plot
        )
        
        print("\nROI Coordinates:")
        for i, (roi_ra, roi_dec) in enumerate(roi_coords):
            print(f"  {i}: ({roi_ra:.6f}, {roi_dec:.6f})")
        
        if not stars_info.empty:
            print("\nSelected Stars:")
            print(stars_info[['source_id', 'ra', 'dec', 'bp_mag', 'distance', 'is_json_target']].to_string())
            
            if args.output:
                stars_info.to_csv(args.output, index=False)
                print(f"\nSaved star information to: {args.output}")
    
    else:
        # Directory mode
        if not args.directory:
            parser.error("--directory is required when not using --coords")
        
        output_csv = args.output if args.output else Path('roi_stars_info.csv')
        
        combined_df = process_json_files(
            args.directory,
            target_epoch=args.target_epoch,
            save_updates=args.save,
            prioritize_json=args.prioritize_json,
            output_csv=output_csv,
            create_plots=args.plot,
            plot_dir=args.plot_dir,
            show_plots=args.show_plot
        )
        
        if not combined_df.empty:
            print(f"\nTotal stars selected across all targets: {len(combined_df)}")


if __name__ == '__main__':
    main()