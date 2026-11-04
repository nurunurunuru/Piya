"""
src/data/downloader.py
Downloads satellite imagery and geospatial data for Dhaka City.
Sources: Google Earth Engine (Sentinel-2, SRTM), OpenStreetMap
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import osmnx as ox
import geopandas as gpd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Google Earth Engine Downloader
# ---------------------------------------------------------------------------
class GEEDownloader:
    """Downloads Sentinel-2 and DEM data via Google Earth Engine."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        self.config = self._load_config(config_path)
        self._init_gee()

    def _load_config(self, path: str) -> Dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _init_gee(self):
        """Authenticate and initialize GEE."""
        try:
            import ee
            key_file = os.getenv("GEE_KEY_FILE")
            if key_file and Path(key_file).exists():
                credentials = ee.ServiceAccountCredentials(
                    email=os.getenv("GEE_SERVICE_ACCOUNT"),
                    key_file=key_file,
                )
                ee.Initialize(credentials)
            else:
                ee.Authenticate()
                ee.Initialize(project=os.getenv("GEE_PROJECT"))
            logger.info("Google Earth Engine initialized successfully.")
        except Exception as e:
            logger.warning(f"GEE initialization failed: {e}. Using offline mode.")
            self._gee_available = False
            return
        self._gee_available = True

    def download_sentinel2(
        self,
        output_dir: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cloud_pct: Optional[int] = None,
    ) -> str:
        """
        Download Sentinel-2 SR composite for Dhaka and export as GeoTIFF.

        Args:
            output_dir: Directory to save the downloaded image.
            start_date: Start date (YYYY-MM-DD).
            end_date:   End date (YYYY-MM-DD).
            cloud_pct:  Max cloud cover percentage.

        Returns:
            Path to downloaded GeoTIFF file.
        """
        if not self._gee_available:
            raise RuntimeError("GEE not available. Check credentials.")

        import ee

        cfg = self.config
        s2_cfg = cfg["sentinel2"]
        bbox = cfg["study_area"]["bbox"]

        start_date = start_date or s2_cfg["start_date"]
        end_date   = end_date   or s2_cfg["end_date"]
        cloud_pct  = cloud_pct  or s2_cfg["cloud_cover_max"]

        region = ee.Geometry.Rectangle([
            bbox["west"], bbox["south"],
            bbox["east"], bbox["north"],
        ])

        collection = (
            ee.ImageCollection(s2_cfg["collection"])
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .select(s2_cfg["bands"])
            .median()
            .divide(s2_cfg["scale"])
        )

        # Add NDVI as extra band
        ndvi = collection.normalizedDifference(["B08", "B04"]).rename("NDVI")
        collection = collection.addBands(ndvi)

        output_path = Path(output_dir) / "sentinel2_dhaka_composite.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        task = ee.batch.Export.image.toDrive(
            image=collection,
            description="dhaka_sentinel2",
            folder="dhaka_height_analysis",
            fileNamePrefix="sentinel2_dhaka_composite",
            region=region,
            scale=s2_cfg["resolution"],
            crs=f"EPSG:{cfg['study_area']['projection'].split(':')[1]}",
            maxPixels=1e13,
        )
        task.start()
        logger.info(f"Sentinel-2 export task started: {task.id}")
        logger.info("Download will appear in your Google Drive → dhaka_height_analysis/")
        return str(output_path)

    def download_dem(self, output_dir: str) -> str:
        """Download SRTM Digital Elevation Model for Dhaka."""
        if not self._gee_available:
            raise RuntimeError("GEE not available.")

        import ee

        cfg = self.config
        bbox = cfg["study_area"]["bbox"]

        region = ee.Geometry.Rectangle([
            bbox["west"], bbox["south"],
            bbox["east"], bbox["north"],
        ])

        dem = (
            ee.Image(cfg["dem"]["collection"])
            .select("elevation")
            .clip(region)
        )

        output_path = Path(output_dir) / "srtm_dem_dhaka.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        task = ee.batch.Export.image.toDrive(
            image=dem,
            description="dhaka_dem",
            folder="dhaka_height_analysis",
            fileNamePrefix="srtm_dem_dhaka",
            region=region,
            scale=cfg["dem"]["resolution"],
            crs="EPSG:4326",
            maxPixels=1e13,
        )
        task.start()
        logger.info(f"DEM export task started: {task.id}")
        return str(output_path)


# ---------------------------------------------------------------------------
# OpenStreetMap Downloader
# ---------------------------------------------------------------------------
class OSMDownloader:
    """Downloads building footprints from OpenStreetMap via OSMnx."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def download_buildings(self, output_dir: str) -> str:
        """
        Download building footprints with height attributes for Dhaka.

        Returns:
            Path to saved GeoPackage (.gpkg) file.
        """
        cfg = self.config
        bbox = cfg["study_area"]["bbox"]

        logger.info("Downloading OSM building footprints for Dhaka...")
        tags = {"building": True}

        gdf = ox.features.features_from_bbox(
            bbox=(
                bbox["south"], bbox["north"],
                bbox["west"], bbox["east"],
            ),
            tags=tags,
        )

        # Keep only polygon features (building footprints)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

        # Extract relevant columns
        keep_cols = ["geometry", "building", "height", "building:levels",
                     "building:material", "name", "addr:street"]
        available = [c for c in keep_cols if c in gdf.columns]
        gdf = gdf[available].copy()

        # Estimate height from levels if height is missing
        if "building:levels" in gdf.columns and "height" in gdf.columns:
            mask = gdf["height"].isna() & gdf["building:levels"].notna()
            gdf.loc[mask, "height"] = (
                gdf.loc[mask, "building:levels"].astype(float) * 3.0
            )

        output_path = Path(output_dir) / "osm_buildings_dhaka.gpkg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GPKG")

        logger.info(f"Saved {len(gdf)} building footprints → {output_path}")
        return str(output_path)

    def download_zones(self, output_dir: str) -> str:
        """Download administrative zone boundaries from OSM."""
        logger.info("Downloading administrative boundaries...")
        gdf = ox.geocode_to_gdf("Dhaka, Bangladesh")
        output_path = Path(output_dir) / "dhaka_admin_boundary.gpkg"
        gdf.to_file(output_path, driver="GPKG")
        logger.info(f"Saved admin boundary → {output_path}")
        return str(output_path)
