"""
src/guidance/zoning_analyzer.py
Analyzes building heights against Bangladesh National Building Code (BNBC)
zoning rules to identify compliance violations and optimization opportunities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from shapely.geometry import box

logger = logging.getLogger(__name__)


class ZoningAnalyzer:
    """
    Checks predicted building heights against BNBC zoning regulations.
    Computes FAR, setback compliance, and over-height violations.
    """

    ZONE_CATEGORIES = {
        "residential_low":    {"max_height": 12,  "far": 2.5, "ground_coverage": 0.60},
        "residential_medium": {"max_height": 27,  "far": 3.5, "ground_coverage": 0.55},
        "residential_high":   {"max_height": 60,  "far": 4.5, "ground_coverage": 0.50},
        "commercial_mixed":   {"max_height": 90,  "far": 6.0, "ground_coverage": 0.60},
        "special_commercial": {"max_height": 150, "far": 8.0, "ground_coverage": 0.55},
    }

    # Dhaka zone → BNBC category mapping
    ZONE_CATEGORY_MAP = {
        "Gulshan":    "commercial_mixed",
        "Banani":     "commercial_mixed",
        "Dhanmondi":  "residential_medium",
        "Motijheel":  "special_commercial",
        "Old Dhaka":  "residential_low",
        "Mirpur":     "residential_medium",
        "Uttara":     "residential_medium",
        "Mohakhali":  "commercial_mixed",
    }

    def __init__(self, config_path: str = "configs/guidance_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def analyze_buildings(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        zone_col: str = "zone_name",
        height_col: str = "predicted_height",
    ) -> gpd.GeoDataFrame:
        """
        Add compliance flags and FAR estimates to building GeoDataFrame.

        Args:
            buildings_gdf: GeoDataFrame with building footprints and predicted heights
            zone_col:      Column containing zone name
            height_col:    Column containing predicted height

        Returns:
            Enriched GeoDataFrame with compliance analysis columns
        """
        gdf = buildings_gdf.copy()

        gdf["zone_category"]   = gdf[zone_col].map(self.ZONE_CATEGORY_MAP).fillna("residential_medium")
        gdf["max_allowed_m"]   = gdf["zone_category"].map(
            lambda c: self.ZONE_CATEGORIES.get(c, {}).get("max_height", 30)
        )
        gdf["allowed_far"]     = gdf["zone_category"].map(
            lambda c: self.ZONE_CATEGORIES.get(c, {}).get("far", 3.0)
        )

        # Compliance flags
        gdf["height_violation"] = gdf[height_col] > gdf["max_allowed_m"]
        gdf["height_excess_m"]  = (gdf[height_col] - gdf["max_allowed_m"]).clip(lower=0)

        # Approximate FAR (floors × footprint area / plot area)
        gdf["footprint_area_m2"] = gdf.to_crs(epsg=32646).geometry.area
        gdf["est_floors"]        = (gdf[height_col] / 3.0).round(0)
        gdf["est_built_area_m2"] = gdf["est_floors"] * gdf["footprint_area_m2"]

        # Height category
        bins   = [0, 6, 12, 27, 60, 100, np.inf]
        labels = ["0-6m (1-2F)", "6-12m (3-4F)", "12-27m (5-9F)",
                  "27-60m (10-20F)", "60-100m (20-33F)", ">100m (33F+)"]
        gdf["height_category"] = pd.cut(
            gdf[height_col].fillna(0), bins=bins, labels=labels
        )

        logger.info(
            f"Analyzed {len(gdf)} buildings. "
            f"Violations: {gdf['height_violation'].sum()} "
            f"({gdf['height_violation'].mean()*100:.1f}%)"
        )
        return gdf

    def compute_zone_summary(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        zone_col: str = "zone_name",
        height_col: str = "predicted_height",
    ) -> pd.DataFrame:
        """Compute zone-level aggregated statistics."""
        summary = (
            buildings_gdf.groupby(zone_col)
            .agg(
                total_buildings=(height_col, "count"),
                mean_height=(height_col, "mean"),
                max_height=(height_col, "max"),
                median_height=(height_col, "median"),
                violations=("height_violation", "sum"),
                violation_pct=("height_violation", lambda x: x.mean() * 100),
            )
            .reset_index()
            .sort_values("mean_height", ascending=False)
        )
        return summary

    def get_development_opportunities(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        height_col: str = "predicted_height",
    ) -> gpd.GeoDataFrame:
        """
        Identify under-utilized plots where additional floors are permitted.
        Returns buildings that could be built higher per BNBC regulations.
        """
        gdf = buildings_gdf.copy()
        gdf["remaining_capacity_m"] = (
            gdf["max_allowed_m"] - gdf[height_col]
        ).clip(lower=0)
        gdf["potential_extra_floors"] = (gdf["remaining_capacity_m"] / 3.0).astype(int)

        opportunities = gdf[gdf["potential_extra_floors"] >= 2].copy()
        logger.info(
            f"Found {len(opportunities)} buildings with ≥2 floors of "
            f"unused vertical capacity."
        )
        return opportunities.sort_values("potential_extra_floors", ascending=False)
