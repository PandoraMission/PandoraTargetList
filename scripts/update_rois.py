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
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import hashlib
from datetime import datetime
import signal

# Suppress benign ERFA warnings about distance overrides
import warnings
from erfa import ErfaWarning
warnings.filterwarnings('ignore', category=ErfaWarning, message='.*distance overridden.*')


# Constants
PLATE_SCALE = 0.789  # arcsec/pixel
DETECTOR_SIZE = 1280  # pixels
TARGET_OFFSET = 0  # arcsec
ROI_SIZE = 50  # pixels
ROI_HALF_DIAGONAL = ROI_SIZE * PLATE_SCALE * np.sqrt(2) / 2  # arcsec
QUERY_RADIUS = (DETECTOR_SIZE * PLATE_SCALE / 2) - TARGET_OFFSET - ROI_HALF_DIAGONAL  # ~217 arcsec
MIN_SEPARATION_PIXELS = np.sqrt(2) * 50  # pixels
MIN_SEPARATION_ARCSEC = MIN_SEPARATION_PIXELS * PLATE_SCALE  # arcsec
PREFERRED_SEPARATION_ARCSEC = 20.0  # arcsec (soft constraint)
BP_MAG_MIN = 7.0
BP_MAG_MAX = 16.5
GAIA_EPOCH = 2016.0  # J2016.0 for Gaia DR3
DEFAULT_TARGET_EPOCH_BJD = 2461222.5  # July 1, 2026 in BJD_TDB
DEDUP_TOLERANCE_ARCSEC = 2.0
DEFAULT_CACHE_FILE = Path('pandora_targets_cache.csv')
DEFAULT_CACHE_INDEX_FILE = Path('pandora_targets_cache_index.json')
GAIA_ONLINE_TIMEOUT = 7.0  # seconds


class TimeoutException(Exception):
    """Exception raised when a query times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Query timed out")


def bjd_tdb_to_jyear(bjd_tdb: float) -> float:
    """Convert BJD_TDB to Julian year (decimal year)."""
    # BJD_TDB is already a full Julian Date (barycentric, TDB scale)
    t = Time(bjd_tdb, format='jd', scale='tdb')
    return t.jyear


def jyear_to_bjd_tdb(jyear: float) -> float:
    """Convert Julian year to BJD_TDB."""
    t = Time(jyear, format='jyear', scale='tdb')
    return t.jd  # Already returns full JD


def apply_proper_motion(ra: float, dec: float, pmra: float, pmdec: float,
                       epoch_start: float, epoch_end: float,
                       parallax: Optional[float] = None,
                       radial_velocity: Optional[float] = None,
                       debug: bool = False) -> Tuple[float, float]:
    """
    Apply proper motion correction to coordinates using astropy.
    
    Parameters:
    -----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    pmra : float
        Proper motion in RA * cos(dec) (mas/yr) - as provided by Gaia
    pmdec : float
        Proper motion in Dec (mas/yr)
    epoch_start : float
        Starting epoch in Julian years
    epoch_end : float
        Target epoch in Julian years
    parallax : Optional[float]
        Parallax in mas (optional, improves accuracy)
    radial_velocity : Optional[float]
        Radial velocity in km/s (optional, improves accuracy)
    
    Returns:
    --------
    Tuple[float, float]
        Corrected (RA, Dec) in degrees
    """
    if debug:
        print(f"\n  DEBUG apply_proper_motion:")
        print(f"    Input: RA={ra:.6f}, Dec={dec:.6f}")
        print(f"    PM: pmra={pmra:.3f} mas/yr, pmdec={pmdec:.3f} mas/yr")
        print(f"    Epochs: {epoch_start:.1f} -> {epoch_end:.1f} ({epoch_end - epoch_start:.1f} years)")
        print(f"    Parallax: {parallax if parallax is not None else 'None'}")
        print(f"    RV: {radial_velocity if radial_velocity is not None else 'None'}")

    # Create SkyCoord object with proper motion
    coord_kwargs = {
        'ra': ra * u.deg,
        'dec': dec * u.deg,
        'pm_ra_cosdec': pmra * u.mas / u.year,  # Note: pm_ra_cosdec, not pm_ra
        'pm_dec': pmdec * u.mas / u.year,
        'obstime': Time(epoch_start, format='jyear'),
        'frame': 'icrs'
    }
    
    # Add distance if parallax is available
    if parallax is not None and not np.isnan(parallax) and parallax > 0:
        try:
            coord_kwargs['distance'] = Distance(parallax=parallax * u.mas, allow_negative=True)
        except:
            pass  # If distance calculation fails, continue without it
    
    # Add radial velocity if available
    if radial_velocity is not None and not np.isnan(radial_velocity):
        coord_kwargs['radial_velocity'] = radial_velocity * u.km / u.s
    
    # Create coordinate object
    coord = SkyCoord(**coord_kwargs)
    
    # Apply space motion to target epoch
    target_time = Time(epoch_end, format='jyear')
    coord_corrected = coord.apply_space_motion(target_time)

    if debug:
        sep = coord.separation(coord_corrected).arcsec
        print(f"    Output: RA={coord_corrected.ra.deg:.6f}, Dec={coord_corrected.dec.deg:.6f}")
        print(f"    Total motion: {sep:.3f} arcsec")
    
    return coord_corrected.ra.deg, coord_corrected.dec.deg


def query_gaia_online(ra: float, dec: float, radius_arcsec: float,
                      bp_mag_min: float = BP_MAG_MIN,
                      bp_mag_max: float = BP_MAG_MAX,
                      timeout: float = GAIA_ONLINE_TIMEOUT) -> Optional[pd.DataFrame]:
    """
    Query Gaia DR3 online archive for stars in a circular region.
    """
    from astroquery.gaia import Gaia
    
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    radius = radius_arcsec * u.arcsec
    
    query = f"""
    SELECT source_id, ra, dec, pmra, pmdec, parallax, radial_velocity,
           phot_bp_mean_mag, phot_g_mean_mag, teff_gspphot
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
        # Set up timeout using signal (Unix-like systems) or threading
        if hasattr(signal, 'SIGALRM'):
            # Unix-like systems
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            try:
                job = Gaia.launch_job(query)
                results = job.get_results()
                signal.alarm(0)  # Cancel the alarm
                return results.to_pandas()
            except TimeoutException:
                signal.alarm(0)
                return None
        else:
            # Windows or systems without SIGALRM - use threading
            from threading import Thread
            result_container = [None]
            exception_container = [None]
            
            def run_query():
                try:
                    job = Gaia.launch_job(query)
                    results = job.get_results()
                    result_container[0] = results.to_pandas()
                except Exception as e:
                    exception_container[0] = e
            
            thread = Thread(target=run_query)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                # Timeout occurred
                return None
            
            if exception_container[0]:
                raise exception_container[0]
            
            return result_container[0]
    
    except Exception as e:
        warnings.warn(f"Gaia online query failed: {e}")
        return None


def query_gaia_offline(ra: float, dec: float, radius_arcsec: float,
                       bp_mag_min: float = BP_MAG_MIN,
                       bp_mag_max: float = BP_MAG_MAX) -> Optional[pd.DataFrame]:
    """
    Query Gaia offline catalog for stars in a circular region.
    """
    try:
        from gaiaoffline import Gaia as GaiaOffline
        
        # Convert radius from arcsec to degrees
        radius_deg = radius_arcsec / 3600.0
        
        with GaiaOffline(magnitude_limit=(bp_mag_min, bp_mag_max), photometry_output='mag') as gaia:
            results = gaia.conesearch(ra=ra, dec=dec, radius=radius_deg)
        
        if results is None or len(results) == 0:
            return pd.DataFrame()
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(results)
        
        # Rename columns to match online Gaia format
        column_mapping = {
            'source_id': 'source_id',
            'ra': 'ra',
            'dec': 'dec',
            'pmra': 'pmra',
            'pmdec': 'pmdec',
            'parallax': 'parallax',
            'radial_velocity': 'radial_velocity',
            'phot_bp_mean_mag': 'phot_bp_mean_mag',
            'phot_g_mean_mag': 'phot_g_mean_mag',
            'teff_gspphot': 'teff_gspphot'
        }
        
        # Check which columns exist and rename them
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df = df.rename(columns={old_col: new_col})
        
        # Filter by Bp magnitude if available
        if 'phot_bp_mean_mag' in df.columns:
            df = df[
                (df['phot_bp_mean_mag'].notna()) &
                (df['phot_bp_mean_mag'] > bp_mag_min) &
                (df['phot_bp_mean_mag'] < bp_mag_max)
            ]
        
        # Filter for required proper motion data
        if 'pmra' in df.columns and 'pmdec' in df.columns:
            df = df[(df['pmra'].notna()) & (df['pmdec'].notna())]
        
        # Sort by Bp magnitude if available
        if 'phot_bp_mean_mag' in df.columns:
            df = df.sort_values('phot_bp_mean_mag')
        
        return df
    
    except ImportError:
        warnings.warn("gaiaoffline package not installed. Install with: pip install gaiaoffline")
        return None
    except Exception as e:
        warnings.warn(f"Gaia offline query failed: {e}")
        return None


def query_gaia_stars(ra: float, dec: float, radius_arcsec: float,
                     bp_mag_min: float = BP_MAG_MIN,
                     bp_mag_max: float = BP_MAG_MAX,
                     use_offline: bool = False,
                     allow_fallback: bool = True) -> pd.DataFrame:
    """
    Query Gaia DR3 for stars in a circular region.
    Tries online first, falls back to offline if online fails or times out.
    
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
    use_offline : bool
        If True, use offline catalog by default (skip online query)
    allow_fallback : bool
        If True, allow fallback to other method if primary fails
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Gaia stars
    """
    result = None
    
    if use_offline:
        # Try offline first
        print(f"    Querying Gaia offline catalog...")
        result = query_gaia_offline(ra, dec, radius_arcsec, bp_mag_min, bp_mag_max)
        
        if result is None or len(result) == 0:
            if allow_fallback:
                print(f"    Offline query failed or empty, falling back to online...")
                result = query_gaia_online(ra, dec, radius_arcsec, bp_mag_min, bp_mag_max)
            else:
                print(f"    Offline query failed or empty, no fallback allowed")
    else:
        # Try online first
        print(f"    Querying Gaia online archive (timeout: {GAIA_ONLINE_TIMEOUT}s)...")
        result = query_gaia_online(ra, dec, radius_arcsec, bp_mag_min, bp_mag_max)
        
        if result is None:
            if allow_fallback:
                print(f"    Online query timed out or failed, falling back to offline catalog...")
                result = query_gaia_offline(ra, dec, radius_arcsec, bp_mag_min, bp_mag_max)
            else:
                print(f"    Online query timed out or failed, no fallback allowed")
    
    # Return empty DataFrame if all queries failed
    if result is None:
        warnings.warn(f"All Gaia queries failed for RA={ra:.4f}, Dec={dec:.4f}")
        return pd.DataFrame()
    
    return result


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
    """
    target_epoch_jyear = bjd_tdb_to_jyear(target_epoch)
    
    # Start with target star
    selected_coords = [(target_ra, target_dec)]
    selected_info = []
    
    # Create candidate list starting with prioritized JSON targets
    candidates = []
    
    if json_targets:
        for json_target in json_targets:
            # Apply proper motion to JSON target WITH parallax/RV from cache
            json_ra, json_dec = apply_proper_motion(
                json_target['ra'], json_target['dec'],
                json_target['pmra'], json_target['pmdec'],
                GAIA_EPOCH, target_epoch_jyear,
                parallax=json_target.get('parallax'),
                radial_velocity=json_target.get('radial_velocity')
            )
            
            # Check if within query radius
            sep = calculate_separation(target_ra, target_dec, json_ra, json_dec)
            if sep <= QUERY_RADIUS and sep > 0.1:
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
            # Apply proper motion with parallax and RV if available
            star_ra, star_dec = apply_proper_motion(
                star['ra'], star['dec'],
                star['pmra'], star['pmdec'],
                GAIA_EPOCH, target_epoch_jyear,
                parallax=star.get('parallax'),
                radial_velocity=star.get('radial_velocity')
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
                    'parallax': star.get('parallax', np.nan),
                    'radial_velocity': star.get('radial_velocity', np.nan),
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


def get_file_mtime(filepath: Path) -> float:
    """Get file modification time."""
    try:
        return filepath.stat().st_mtime
    except:
        return 0.0


def load_json_file_with_info(filepath: Path) -> Tuple[Path, Optional[Dict], Optional[str], float]:
    """
    Load JSON file and return path, data, content hash, and modification time.
    Helper function for parallel processing.
    """
    data = load_json_file(filepath)
    file_hash = compute_file_hash(filepath) if data else None
    mtime = get_file_mtime(filepath)
    return filepath, data, file_hash, mtime


def load_cache_index(index_file: Path) -> Dict:
    """
    Load cache index containing file metadata.
    
    Parameters:
    -----------
    index_file : Path
        Path to cache index JSON file
    
    Returns:
    --------
    Dict
        Cache index with file metadata
    """
    if not index_file.exists():
        return {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'files': {}
        }
    
    try:
        with open(index_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Error loading cache index: {e}")
        return {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'files': {}
        }


def save_cache_index(index_data: Dict, index_file: Path):
    """
    Save cache index to file.
    
    Parameters:
    -----------
    index_data : Dict
        Cache index data
    index_file : Path
        Path to cache index JSON file
    """
    try:
        index_data['last_updated'] = datetime.now().isoformat()
        index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
    except Exception as e:
        warnings.warn(f"Error saving cache index: {e}")


def get_cache_metadata(directories: List[Path]) -> Dict:
    """
    Generate metadata about directories for cache validation.
    
    Parameters:
    -----------
    directories : List[Path]
        List of directory paths
    
    Returns:
    --------
    Dict
        Metadata including directory paths, file count, and last modified times
    """
    metadata = {
        'directories': [str(d.resolve()) for d in directories],
        'directory_count': len(directories),
        'total_json_files': 0,
        'last_modified': None
    }
    
    # Get file count and latest modification time
    latest_mtime = 0
    for directory in directories:
        for json_file in directory.rglob("*.json"):
            metadata['total_json_files'] += 1
            mtime = get_file_mtime(json_file)
            if mtime > latest_mtime:
                latest_mtime = mtime
    
    if latest_mtime > 0:
        metadata['last_modified'] = datetime.fromtimestamp(latest_mtime).isoformat()
    
    return metadata


def save_targets_cache(targets: List[Dict], cache_file: Path, directories: List[Path]):
    """
    Save deduplicated targets to cache file.
    
    Parameters:
    -----------
    targets : List[Dict]
        List of target dictionaries
    cache_file : Path
        Path to save cache CSV
    directories : List[Path]
        Directories that were scanned
    """
    if not targets:
        print("No targets to cache")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(targets)
    
    # Remove source_file column if it exists (internal use only)
    if 'source_file' in df.columns:
        df = df.drop(columns=['source_file'])
    
    # Add metadata as attributes (won't be saved to CSV, but useful for in-memory)
    metadata = get_cache_metadata(directories)
    
    # Save to CSV with metadata in header comments
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_file, 'w') as f:
        # Write metadata as comments
        f.write(f"# Pandora Targets Cache\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Directories: {', '.join(metadata['directories'])}\n")
        f.write(f"# Total JSON files scanned: {metadata['total_json_files']}\n")
        f.write(f"# Unique targets: {len(targets)}\n")
        f.write(f"# Last modified: {metadata.get('last_modified', 'N/A')}\n")
        f.write("#\n")
        
        # Write CSV data
        df.to_csv(f, index=False)
    
    print(f"Saved {len(targets)} targets to cache: {cache_file}")


def load_targets_cache(cache_file: Path) -> List[Dict]:
    """
    Load targets from cache file.
    
    Parameters:
    -----------
    cache_file : Path
        Path to cache CSV file
    
    Returns:
    --------
    List[Dict]
        List of target dictionaries
    """
    try:
        df = pd.read_csv(cache_file, comment='#')
        
        # Convert NaN to None for proper JSON serialization
        df = df.where(pd.notnull(df), None)
        
        # Convert to list of dicts
        targets = df.to_dict('records')
        
        return targets
        
    except Exception as e:
        warnings.warn(f"Error loading cache: {e}")
        return []


def collect_json_targets_incremental(directories: List[Path], 
                                     cache_file: Path,
                                     index_file: Path,
                                     use_parallel: bool = True,
                                     force_rebuild: bool = False,
                                     use_gaia_offline: bool = False) -> List[Dict]:
    """
    Collect target information with incremental cache updates.
    Queries Gaia for parallax and radial velocity for each target.
    
    Parameters:
    -----------
    directories : List[Path]
        List of directory paths to search
    cache_file : Path
        Path to cache CSV file
    index_file : Path
        Path to cache index JSON file
    use_parallel : bool
        Whether to use parallel processing (default True)
    force_rebuild : bool
        Force rebuilding entire cache
    use_gaia_offline : bool
        Use offline Gaia catalog for queries
    
    Returns:
    --------
    List[Dict]
        List of target dictionaries with deduplicated coordinates
    """
    # Load cache index
    cache_index = load_cache_index(index_file)
    
    # Load existing cache if available
    existing_targets = {}
    if cache_file.exists() and not force_rebuild:
        cached = load_targets_cache(cache_file)
        # Index by coordinates for quick lookup
        for target in cached:
            key = (target['ra'], target['dec'])
            existing_targets[key] = target
        print(f"Loaded {len(existing_targets)} targets from existing cache")
    
    # Scan all JSON files
    print("Scanning directories for JSON files...")
    json_files = []
    for directory in directories:
        json_files.extend(directory.rglob("*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Determine which files need to be processed
    files_to_process = []
    files_unchanged = 0
    files_new = 0
    files_modified = 0
    files_duplicate_skipped = 0
    
    for json_file in json_files:
        file_path_str = str(json_file.resolve())
        current_mtime = get_file_mtime(json_file)
        
        if force_rebuild:
            files_to_process.append(json_file)
        elif file_path_str in cache_index['files']:
            cached_entry = cache_index['files'][file_path_str]
            
            # Check if file has been modified
            cached_mtime = cached_entry.get('mtime', 0)
            if current_mtime > cached_mtime:
                files_to_process.append(json_file)
                files_modified += 1
            else:
                # File unchanged - check if it's marked as duplicate
                if cached_entry.get('is_duplicate', False):
                    files_duplicate_skipped += 1
                else:
                    files_unchanged += 1
        else:
            # New file not in cache
            files_to_process.append(json_file)
            files_new += 1
    
    if not force_rebuild:
        print(f"Cache status: {files_unchanged} unchanged, {files_duplicate_skipped} duplicate, "
              f"{files_modified} modified, {files_new} new")
        print(f"Processing {len(files_to_process)} files...")
    else:
        print(f"Force rebuild: Processing all {len(files_to_process)} files...")
    
    # Process files that need updating
    seen_hashes = set()
    seen_coords = list(existing_targets.keys())
    targets = list(existing_targets.values())
    
    # Track which existing targets to keep (those not from modified files)
    modified_file_paths = {str(f.resolve()) for f in files_to_process}
    
    # Only remove targets that came from files we're re-processing
    targets_from_unchanged = []
    for coord_key, target in existing_targets.items():
        keep_target = True
        
        # Check all files in cache index that point to this coordinate
        for file_path, file_info in cache_index['files'].items():
            if file_info.get('coordinates') == list(coord_key):
                if file_path in modified_file_paths:
                    keep_target = False
                    break
        
        if keep_target:
            targets_from_unchanged.append(target)
    
    # Start fresh with targets from unchanged files
    targets = targets_from_unchanged
    seen_coords = [(t['ra'], t['dec']) for t in targets]
    
    targets_updated = 0
    targets_added = 0
    duplicates_found = 0
    gaia_queries = 0
    
    if files_to_process:
        # Load files in parallel if requested
        if use_parallel and len(files_to_process) > 10:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(load_json_file_with_info, f): f for f in files_to_process}
                
                for i, future in enumerate(as_completed(futures), 1):
                    if i % 100 == 0:
                        print(f"  Processed {i}/{len(files_to_process)} files...")
                    
                    try:
                        filepath, data, file_hash, mtime = future.result()
                        file_path_str = str(filepath.resolve())
                        
                        if not data or 'RA' not in data or 'DEC' not in data:
                            continue
                        
                        ra = data['RA']
                        dec = data['DEC']
                        
                        # Check if this is a duplicate by content hash
                        is_content_duplicate = file_hash and file_hash in seen_hashes
                        
                        # Check if this is a duplicate by coordinates
                        is_coord_duplicate = False
                        for seen_ra, seen_dec in seen_coords:
                            if calculate_separation(ra, dec, seen_ra, seen_dec) < DEDUP_TOLERANCE_ARCSEC:
                                is_coord_duplicate = True
                                break
                        
                        is_duplicate = is_content_duplicate or is_coord_duplicate
                        
                        # Always update cache index, even for duplicates
                        cache_index['files'][file_path_str] = {
                            'mtime': mtime,
                            'hash': file_hash,
                            'star_name': data.get('Star Name', 'Unknown'),
                            'coordinates': [ra, dec],
                            'is_duplicate': is_duplicate
                        }
                        
                        if is_duplicate:
                            duplicates_found += 1
                            continue
                        
                        # Not a duplicate - query Gaia for additional data
                        pmra = data.get('pm_RA', 0.0)
                        pmdec = data.get('pm_DEC', 0.0)
                        
                        # Query Gaia for parallax and radial velocity
                        parallax = np.nan
                        radial_velocity = np.nan
                        
                        # Small search radius around the target (5 arcsec)
                        gaia_result = query_gaia_stars(ra, dec, radius_arcsec=5.0, 
                                                      use_offline=use_gaia_offline,
                                                      allow_fallback=True)
                        
                        if not gaia_result.empty:
                            gaia_queries += 1
                            # Find closest match to target coordinates
                            min_sep = float('inf')
                            best_match = None
                            
                            for _, star in gaia_result.iterrows():
                                sep = calculate_separation(ra, dec, star['ra'], star['dec'])
                                if sep < min_sep:
                                    min_sep = sep
                                    best_match = star
                            
                            # If match is within 2 arcsec, use it
                            if best_match is not None and min_sep < 2.0:
                                parallax = best_match.get('parallax', np.nan)
                                radial_velocity = best_match.get('radial_velocity', np.nan)
                        
                        # Add to targets with Gaia data
                        target = {
                            'ra': ra,
                            'dec': dec,
                            'pmra': pmra,
                            'pmdec': pmdec,
                            'parallax': parallax,
                            'radial_velocity': radial_velocity,
                            'Bmag': data.get('Bmag', np.nan),
                            'Gmag': data.get('Gmag', np.nan),
                            'Teff (K)': data.get('Teff (K)', np.nan),
                            'Star Name': data.get('Star Name', 'Unknown'),
                            'source_file': file_path_str
                        }
                        targets.append(target)
                        seen_coords.append((ra, dec))
                        
                        if file_hash:
                            seen_hashes.add(file_hash)
                        
                        if file_path_str in cache_index.get('files', {}):
                            targets_updated += 1
                        else:
                            targets_added += 1
                    
                    except Exception as e:
                        warnings.warn(f"Error processing file: {e}")
        else:
            # Sequential processing
            for i, json_file in enumerate(files_to_process, 1):
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(files_to_process)} files...")
                
                data = load_json_file(json_file)
                file_path_str = str(json_file.resolve())
                
                if not data or 'RA' not in data or 'DEC' not in data:
                    continue
                
                file_hash = compute_file_hash(json_file)
                mtime = get_file_mtime(json_file)
                
                ra = data['RA']
                dec = data['DEC']
                
                # Check if this is a duplicate by content hash
                is_content_duplicate = file_hash and file_hash in seen_hashes
                
                # Check if this is a duplicate by coordinates
                is_coord_duplicate = False
                for seen_ra, seen_dec in seen_coords:
                    if calculate_separation(ra, dec, seen_ra, seen_dec) < DEDUP_TOLERANCE_ARCSEC:
                        is_coord_duplicate = True
                        break
                
                is_duplicate = is_content_duplicate or is_coord_duplicate
                
                # Always update cache index, even for duplicates
                cache_index['files'][file_path_str] = {
                    'mtime': mtime,
                    'hash': file_hash,
                    'star_name': data.get('Star Name', 'Unknown'),
                    'coordinates': [ra, dec],
                    'is_duplicate': is_duplicate
                }
                
                if is_duplicate:
                    duplicates_found += 1
                    continue
                
                # Not a duplicate - query Gaia for additional data
                pmra = data.get('pm_RA', 0.0)
                pmdec = data.get('pm_DEC', 0.0)
                
                # Query Gaia for parallax and radial velocity
                parallax = np.nan
                radial_velocity = np.nan
                
                print(f"  Querying Gaia for {data.get('Star Name', 'Unknown')}...")
                # Small search radius around the target (5 arcsec)
                gaia_result = query_gaia_stars(ra, dec, radius_arcsec=5.0, 
                                              use_offline=use_gaia_offline,
                                              allow_fallback=True)
                
                if not gaia_result.empty:
                    gaia_queries += 1
                    # Find closest match to target coordinates
                    min_sep = float('inf')
                    best_match = None
                    
                    for _, star in gaia_result.iterrows():
                        sep = calculate_separation(ra, dec, star['ra'], star['dec'])
                        if sep < min_sep:
                            min_sep = sep
                            best_match = star
                    
                    # If match is within 2 arcsec, use it
                    if best_match is not None and min_sep < 2.0:
                        parallax = best_match.get('parallax', np.nan)
                        radial_velocity = best_match.get('radial_velocity', np.nan)
                        print(f"    Found Gaia match: parallax={parallax:.2f} mas, RV={radial_velocity:.2f} km/s")
                    else:
                        print(f"    No close Gaia match found (closest: {min_sep:.2f} arcsec)")
                
                # Add to targets with Gaia data
                target = {
                    'ra': ra,
                    'dec': dec,
                    'pmra': pmra,
                    'pmdec': pmdec,
                    'parallax': parallax,
                    'radial_velocity': radial_velocity,
                    'Bmag': data.get('Bmag', np.nan),
                    'Gmag': data.get('Gmag', np.nan),
                    'Teff (K)': data.get('Teff (K)', np.nan),
                    'Star Name': data.get('Star Name', 'Unknown'),
                    'source_file': file_path_str
                }
                targets.append(target)
                seen_coords.append((ra, dec))
                
                if file_hash:
                    seen_hashes.add(file_hash)
                
                if file_path_str in cache_index.get('files', {}):
                    targets_updated += 1
                else:
                    targets_added += 1
    
    print(f"Cache update complete: {len(targets)} unique targets")
    if not force_rebuild:
        print(f"  From cache: {len(targets_from_unchanged)}, "
              f"Updated: {targets_updated}, Added: {targets_added}, Duplicates: {duplicates_found}")
    print(f"  Gaia queries performed: {gaia_queries}")
    
    # Save updated cache
    save_targets_cache(targets, cache_file, directories)
    save_cache_index(cache_index, index_file)
    
    return targets


def collect_json_targets(directories: List[Path], 
                        use_parallel: bool = True,
                        cache_file: Optional[Path] = None,
                        index_file: Optional[Path] = None,
                        force_rebuild: bool = False,
                        use_gaia_offline: bool = False) -> List[Dict]:
    """
    Collect target information from all JSON files in directories.
    Uses incremental caching if cache files are provided.
    
    Parameters:
    -----------
    directories : List[Path]
        List of directory paths to search
    use_parallel : bool
        Whether to use parallel processing (default True)
    cache_file : Optional[Path]
        Path to cache file (if None, caching is disabled)
    index_file : Optional[Path]
        Path to cache index file (if None, uses default based on cache_file)
    force_rebuild : bool
        Force rebuilding cache even if valid cache exists
    use_gaia_offline : bool
        Use offline Gaia catalog for queries
    
    Returns:
    --------
    List[Dict]
        List of target dictionaries with deduplicated coordinates
    """
    # If caching is enabled, use incremental approach
    if cache_file:
        if index_file is None:
            # Derive index file name from cache file
            index_file = cache_file.parent / (cache_file.stem + '_index.json')
        
        return collect_json_targets_incremental(
            directories, cache_file, index_file, use_parallel, force_rebuild, use_gaia_offline
        )
    
    # Otherwise, use original non-cached approach
    print("Caching disabled, scanning all files...")
    print("Scanning directories for JSON files...")
    
    # Collect all JSON file paths
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
                    filepath, data, file_hash, _ = future.result()
                    
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
                      show_plots: bool = False,
                      cache_file: Optional[Path] = None,
                      index_file: Optional[Path] = None,
                      rebuild_cache: bool = False,
                      use_gaia_offline: bool = False,
                      process_all: bool = False) -> pd.DataFrame:
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
    cache_file : Optional[Path]
        Path to cache file for JSON targets
    index_file : Optional[Path]
        Path to cache index file
    rebuild_cache : bool
        Force rebuilding of cache
    use_gaia_offline : bool
        Use offline Gaia catalog by default
    process_all : bool
        Process all JSON files with RA/DEC, even without existing ROI_coord
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all selected stars
    """
    all_stars_info = []
    json_targets = None
    
    if prioritize_json:
        json_targets = collect_json_targets(
            directories, 
            cache_file=cache_file,
            index_file=index_file,
            force_rebuild=rebuild_cache,
            use_gaia_offline=use_gaia_offline
        )
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
            
            if not data:
                continue
            
            # Skip if missing required coordinate fields
            if 'RA' not in data or 'DEC' not in data:
                continue
            
            # Check if we should process this file
            if not process_all and 'ROI_coord' not in data:
                # Skip files without ROI_coord unless --process-all is specified
                continue
            
            # If ROI_coord doesn't exist but we're processing all, create it
            if 'ROI_coord' not in data:
                data['ROI_coord'] = []
                data['numPredefinedStarRois'] = 0
                print(f"\nProcessing NEW target (no existing ROI_coord): {json_file.name}")
            
            files_processed += 1
            
            # Extract target information
            target_ra_orig = data['RA']
            target_dec_orig = data['DEC']
            target_pmra = data.get('pm_RA', 0.0)
            target_pmdec = data.get('pm_DEC', 0.0)
            coord_epoch_str = data.get('Coordinate Epoch', 'J2016.0')

            # Parse epoch from JSON if available
            if coord_epoch_str.startswith('J'):
                json_epoch = float(coord_epoch_str[1:])
            else:
                json_epoch = GAIA_EPOCH
                warnings.warn(f"Could not parse epoch '{coord_epoch_str}', assuming J{GAIA_EPOCH}")

            print(f"  Target coordinates at epoch {coord_epoch_str}: ({target_ra_orig:.6f}, {target_dec_orig:.6f})")
            print(f"  Proper motion: pmRA={target_pmra:.3f}, pmDec={target_pmdec:.3f} mas/yr")

            # Query Gaia for target's parallax and radial velocity
            target_parallax = np.nan
            target_radial_velocity = np.nan

            print(f"  Querying Gaia for target astrometry...")
            gaia_target_result = query_gaia_stars(target_ra_orig, target_dec_orig, 
                                                radius_arcsec=5.0, 
                                                use_offline=use_gaia_offline,
                                                allow_fallback=True)

            if not gaia_target_result.empty:
                # Find closest match
                min_sep = float('inf')
                best_match = None
                
                for _, star in gaia_target_result.iterrows():
                    sep = calculate_separation(target_ra_orig, target_dec_orig, 
                                            star['ra'], star['dec'])
                    if sep < min_sep:
                        min_sep = sep
                        best_match = star
                
                if best_match is not None and min_sep < 2.0:
                    target_parallax = best_match.get('parallax', np.nan)
                    target_radial_velocity = best_match.get('radial_velocity', np.nan)

            # Apply proper motion to target WITH parallax/RV
            target_ra, target_dec = apply_proper_motion(
                target_ra_orig, target_dec_orig,
                target_pmra, target_pmdec,
                json_epoch, target_epoch_jyear,
                parallax=target_parallax,
                radial_velocity=target_radial_velocity,
                debug=False
            )

            print(f"  Target after PM correction to {target_epoch_jyear:.1f}: ({target_ra:.6f}, {target_dec:.6f})")
            
            target_name = data.get('Star Name', 'Unknown')
            print(f"\nProcessing: {json_file.name}")
            print(f"  Target: {target_name} at ({target_ra:.6f}, {target_dec:.6f})")
            
            # Query Gaia
            gaia_stars = query_gaia_stars(target_ra, target_dec, QUERY_RADIUS, 
                                         use_offline=use_gaia_offline)
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
            data['ROI_coord_epoch'] = 'J2026.5'

            # Set StarRoiDetMethod to 1 when ROI coordinates are selected
            # Only update if we actually selected coordinates
            if len(roi_coords) > 0:
                data['StarRoiDetMethod'] = 1
            
            # Save if requested
            if save_updates:
                if save_json_file(json_file, data):
                    files_updated += 1
                    print(f"  ✓ Updated {json_file}")
                    print(f"    - ROI_coord: {len(roi_coords)} coordinate pairs")
                    print(f"    - numPredefinedStarRois: {data['numPredefinedStarRois']}")
                    print(f"    - StarRoiDetMethod: {data['StarRoiDetMethod']}")
            
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
                'pmra', 'pmdec', 'parallax', 'radial_velocity',
                'bp_mag', 'g_mag', 'teff',
                'distance', 'is_json_target']
        
        # Only include columns that exist
        cols = [c for c in cols if c in combined_df.columns]
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
                       show_plot: bool = False,
                       cache_file: Optional[Path] = None,
                       index_file: Optional[Path] = None,
                       rebuild_cache: bool = False,
                       use_gaia_offline: bool = False) -> Tuple[List[Tuple[float, float]], pd.DataFrame]:
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
    cache_file : Optional[Path]
        Path to cache file for JSON targets
    index_file : Optional[Path]
        Path to cache index file
    rebuild_cache : bool
        Force rebuilding of cache
    use_gaia_offline : bool
        Use offline Gaia catalog by default
    
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
        json_targets = collect_json_targets(
            json_directories,
            cache_file=cache_file,
            index_file=index_file,
            force_rebuild=rebuild_cache,
            use_gaia_offline=use_gaia_offline
        )
        print(f"Found {len(json_targets)} unique JSON targets\n")
    
    # Query Gaia
    print(f"Querying Gaia within {QUERY_RADIUS:.1f} arcsec...")
    gaia_stars = query_gaia_stars(target_ra, target_dec, QUERY_RADIUS, 
                                  use_offline=use_gaia_offline)
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
  # Process all JSON files in a directory (dry run, uses cache)
  python roi_updater.py --directory /path/to/targets
  
  # Process ALL JSON files, even those without existing ROI_coord field
  python roi_updater.py --directory /path/to/targets --process-all --save

  # Process and save updates with diagnostic plots
  python roi_updater.py --directory /path/to/targets --save --plot --plot-dir ./plots
  
  # Process with JSON target prioritization (creates/uses cache)
  python roi_updater.py --directory /path/to/targets --prioritize-json --save
  
  # Use offline Gaia catalog (no online queries)
  python roi_updater.py --directory /path/to/targets --use-gaia-offline --save
  
  # Force rebuild of cache
  python roi_updater.py --directory /path/to/targets --prioritize-json --rebuild-cache
  
  # Use custom cache location
  python roi_updater.py -d /path/to/targets --prioritize-json --cache-file /path/to/my_cache.csv
  
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
    parser.add_argument('--process-all', action='store_true',  # ← ADD THIS
                       help='Process all JSON files with RA/DEC, even those without existing ROI_coord field')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output CSV file for star information')
    
    # Plotting options
    parser.add_argument('--plot', action='store_true',
                       help='Create diagnostic plots showing FOV and ROI positions')
    parser.add_argument('--plot-dir', type=Path,
                       help='Directory to save diagnostic plots (default: ./roi_plots)')
    parser.add_argument('--show-plot', action='store_true',
                       help='Display plots interactively (default: save only)')
    
    # Caching options
    parser.add_argument('--cache-file', type=Path, default=DEFAULT_CACHE_FILE,
                       help=f'Path to cache file for JSON targets (default: {DEFAULT_CACHE_FILE})')
    parser.add_argument('--cache-index', type=Path,
                       help='Path to cache index file (default: derived from cache-file)')
    parser.add_argument('--rebuild-cache', action='store_true',
                       help='Force rebuilding of target cache')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching (slower but ensures fresh data)')
    
    # Gaia options
    parser.add_argument('--use-gaia-offline', action='store_true',
                       help='Use offline Gaia catalog (gaiaoffline package) instead of online queries')
    
    args = parser.parse_args()
    
    # Validate proper motion arguments
    if args.pm and not args.epoch:
        parser.error("--pm requires --epoch to be specified")
    if args.epoch and not args.pm:
        parser.error("--epoch requires --pm to be specified")
    
    # Set default plot directory
    if args.plot and not args.plot_dir:
        args.plot_dir = Path('./roi_plots')
    
    # Handle cache file
    cache_file = None if args.no_cache else args.cache_file
    index_file = args.cache_index
    
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
            show_plot=args.show_plot,
            cache_file=cache_file,
            index_file=index_file,
            rebuild_cache=args.rebuild_cache,
            use_gaia_offline=args.use_gaia_offline
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
            show_plots=args.show_plot,
            cache_file=cache_file,
            index_file=index_file,
            rebuild_cache=args.rebuild_cache,
            use_gaia_offline=args.use_gaia_offline,
            process_all=args.process_all
        )
        
        if not combined_df.empty:
            print(f"\nTotal stars selected across all targets: {len(combined_df)}")


if __name__ == '__main__':
    main()