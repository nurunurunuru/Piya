"""
pipeline/stage_01_ingest.py
Stage 1: Download Sentinel-2 imagery, DEM, and OSM building footprints.
"""
import logging
from pathlib import Path
logger = logging.getLogger(__name__)


def run(config: dict):
    """Download all raw data sources."""
    from src.data.downloader import GEEDownloader, OSMDownloader

    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")

    # --- OSM Buildings (no auth required) ---
    logger.info("Downloading OSM building footprints...")
    osm = OSMDownloader()
    osm.download_buildings(output_dir=raw_dir + "/osm")
    osm.download_zones(output_dir=raw_dir + "/osm")

    if config.get("skip_download"):
        logger.info("Skipping GEE download (--skip-download flag set).")
        return

    # --- Sentinel-2 + DEM via GEE ---
    logger.info("Initiating Google Earth Engine downloads...")
    gee = GEEDownloader()
    gee.download_sentinel2(output_dir=raw_dir + "/satellite")
    gee.download_dem(output_dir=raw_dir + "/lidar")

    logger.info(
        "GEE exports are running. Download from your Google Drive when complete.\n"
        "Move files to data/raw/satellite/ and data/raw/lidar/ then re-run from Stage 2."
    )
