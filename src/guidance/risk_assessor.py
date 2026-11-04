"""
src/guidance/risk_assessor.py
Assesses structural, seismic, flood, and wind risks for Dhaka buildings.
Integrates with height predictions and known hazard zones.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)


class RiskAssessor:
    """
    Multi-hazard risk assessment for Dhaka City buildings.
    Combines predicted heights with hazard data to generate risk scores.
    """

    # Dhaka's high-risk flood zones (per BWDB data)
    HIGH_FLOOD_ZONES = ["Old Dhaka", "Mirpur", "Demra", "Badda"]

    # Bangladesh Seismic Zone II — moderate hazard
    SEISMIC_PGA = 0.15    # g
    SEISMIC_HIGH_RISK_HEIGHT = 60.0   # m — special engineering required

    def assess_structural_risk(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        height_col: str = "predicted_height",
    ) -> gpd.GeoDataFrame:
        """
        Flag buildings at structural risk based on:
        1. Height exceeding zoning by >20% (overloading)
        2. Heights >60m in seismic zone without special design (assumed)
        3. Very tall buildings on soft soil (flood zones = soft soil proxy)
        """
        gdf = buildings_gdf.copy()

        # Structural overloading risk
        gdf["structural_overload_risk"] = (
            gdf[height_col] > gdf.get("max_allowed_m", 30) * 1.2
        )

        # Seismic risk
        gdf["seismic_risk"] = gdf[height_col] > self.SEISMIC_HIGH_RISK_HEIGHT

        # Flood zone risk
        if "zone_name" in gdf.columns:
            gdf["flood_risk"] = gdf["zone_name"].isin(self.HIGH_FLOOD_ZONES)
        else:
            gdf["flood_risk"] = False

        # Combined risk score [0–3]
        gdf["risk_score"] = (
            gdf["structural_overload_risk"].astype(int)
            + gdf["seismic_risk"].astype(int)
            + gdf["flood_risk"].astype(int)
        )

        gdf["risk_level"] = pd.cut(
            gdf["risk_score"],
            bins=[-1, 0, 1, 2, 3],
            labels=["Low", "Moderate", "High", "Critical"],
        )

        logger.info(f"Risk assessment complete:")
        if "risk_level" in gdf.columns:
            logger.info(gdf["risk_level"].value_counts().to_string())
        return gdf

    def generate_risk_matrix(self, buildings_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate a summary risk matrix by zone and risk level."""
        if "risk_level" not in buildings_gdf.columns:
            buildings_gdf = self.assess_structural_risk(buildings_gdf)

        zone_col = "zone_name" if "zone_name" in buildings_gdf.columns else "zone"
        if zone_col not in buildings_gdf.columns:
            buildings_gdf[zone_col] = "Unknown"

        matrix = pd.crosstab(
            buildings_gdf[zone_col],
            buildings_gdf["risk_level"],
            margins=True,
        )
        return matrix

    def get_critical_buildings(
        self, buildings_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Return only buildings with Critical or High risk."""
        if "risk_level" not in buildings_gdf.columns:
            buildings_gdf = self.assess_structural_risk(buildings_gdf)
        return buildings_gdf[
            buildings_gdf["risk_level"].isin(["Critical", "High"])
        ].copy()
