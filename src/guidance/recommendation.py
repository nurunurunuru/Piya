"""
src/guidance/recommendation.py
AI-driven urban planning recommendations for Dhaka's Modish City initiative.
Combines zoning analysis, risk assessment, and density data to generate
actionable, zone-specific guidance.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)


@dataclass
class ZoneRecommendation:
    zone_name: str
    priority: str           # Critical / High / Medium / Low
    category: str           # Height / Risk / Green / Density / Infrastructure
    title: str
    description: str
    action_items: List[str] = field(default_factory=list)
    estimated_impact: str = ""


class ModishCityRecommender:
    """
    Generates zone-specific Modish City planning recommendations using
    rule-based expert system fed by ML height predictions and risk data.
    """

    HEIGHT_CATEGORIES = {
        "under_utilized": ("Mean height < 6m but zoning allows ≥12m",
                           "Residential Low"),
        "mid_rise_zone":  ("Mean height 6–27m", "Medium Density"),
        "high_rise_zone": ("Mean height > 27m", "High Density Commercial"),
        "overshoot":      ("Buildings exceed max zoning height",
                           "Compliance Risk"),
    }

    def generate(
        self,
        zone_summary: pd.DataFrame,
        risk_matrix: pd.DataFrame,
        buildings_gdf: gpd.GeoDataFrame,
    ) -> List[ZoneRecommendation]:
        """
        Generate a list of planning recommendations for all Dhaka zones.
        """
        recommendations = []

        for _, row in zone_summary.iterrows():
            zone = row.get("zone_name", row.get("zone", "Unknown"))
            mean_h = row.get("mean_height", 0)
            violation_pct = row.get("violation_pct", 0)
            total = row.get("total_buildings", 1)

            # --- Height underutilization ---
            if mean_h < 6.0:
                recommendations.append(ZoneRecommendation(
                    zone_name=zone,
                    priority="Medium",
                    category="Height",
                    title=f"{zone}: Significant Underutilization of Vertical Space",
                    description=(
                        f"Mean building height is only {mean_h:.1f}m. "
                        f"BNBC zoning permits significantly taller structures. "
                        f"The zone has considerable unused FAR capacity."
                    ),
                    action_items=[
                        "Conduct feasibility study for vertical densification",
                        "Provide incentives for mixed-use mid-rise redevelopment",
                        "Ensure adequate infrastructure (water, power) before upzoning",
                        "Preserve heritage buildings in Old Dhaka sub-zones",
                    ],
                    estimated_impact="Could increase housing supply by 40–60% without horizontal sprawl",
                ))

            # --- Height violation ---
            if violation_pct > 10:
                recommendations.append(ZoneRecommendation(
                    zone_name=zone,
                    priority="Critical" if violation_pct > 30 else "High",
                    category="Risk",
                    title=f"{zone}: {violation_pct:.0f}% of Buildings Exceed Zoning Height Limits",
                    description=(
                        f"{violation_pct:.0f}% of {total} buildings exceed BNBC "
                        f"maximum height for this zone. Immediate regulatory review required."
                    ),
                    action_items=[
                        "Conduct structural safety audit of over-height buildings",
                        "Issue stop-work orders for ongoing non-compliant constructions",
                        "Review and update zoning maps to reflect current urban context",
                        "Establish retroactive compliance pathway for existing structures",
                    ],
                    estimated_impact="Reduces seismic and structural collapse risk",
                ))

            # --- High density with no green space ---
            if mean_h > 20:
                recommendations.append(ZoneRecommendation(
                    zone_name=zone,
                    priority="Medium",
                    category="Green Space",
                    title=f"{zone}: High-Density Zone Requires Green Infrastructure",
                    description=(
                        f"Zone {zone} has high average building height ({mean_h:.1f}m). "
                        f"WHO recommends ≥9m² green space per person — Dhaka averages <3m²."
                    ),
                    action_items=[
                        "Mandate rooftop gardens for buildings >10 floors",
                        "Convert underused surface parking to pocket parks",
                        "Introduce green building certification incentives",
                        "Plan linear parks along waterways and road medians",
                    ],
                    estimated_impact="Improves urban heat island, air quality, and resident well-being",
                ))

        # --- City-wide Modish City recommendations ---
        recommendations += self._global_recommendations()
        return sorted(recommendations, key=lambda r: {"Critical": 0, "High": 1,
                                                       "Medium": 2, "Low": 3}[r.priority])

    def _global_recommendations(self) -> List[ZoneRecommendation]:
        """City-wide strategic recommendations for Dhaka Modish City."""
        return [
            ZoneRecommendation(
                zone_name="City-Wide",
                priority="High",
                category="Infrastructure",
                title="Develop Transit-Oriented Development (TOD) Corridors",
                description=(
                    "Concentrate high-rise development within 500m of metro stations "
                    "(MRT Lines 1, 5, 6). Reduces car dependency and enables vertical growth."
                ),
                action_items=[
                    "Designate TOD zones around all 20 MRT stations",
                    "Allow FAR bonus of 1.5× within TOD zones",
                    "Require ground-floor retail and active frontages",
                    "Mandate underground parking to free street-level space",
                ],
                estimated_impact="Reduces traffic congestion by 25%, enables 300,000 new housing units",
            ),
            ZoneRecommendation(
                zone_name="City-Wide",
                priority="High",
                category="Height",
                title="Establish Tiered Height Zoning Based on Infrastructure Capacity",
                description=(
                    "Current zoning is outdated. New height tiers should be tied to "
                    "verified water supply, sewage capacity, and road width."
                ),
                action_items=[
                    "Commission infrastructure capacity study for each zone",
                    "Create GIS-based dynamic height limit map",
                    "Update RAJUK development control regulations",
                    "Introduce conditional height bonuses for affordable housing inclusion",
                ],
                estimated_impact="More equitable and sustainable vertical growth",
            ),
            ZoneRecommendation(
                zone_name="City-Wide",
                priority="Medium",
                category="Risk",
                title="Mandatory Seismic Retrofitting Program for Pre-2006 Buildings",
                description=(
                    "Many buildings pre-date Bangladesh's 2006 seismic code update. "
                    "With Dhaka in Seismic Zone II, structural upgrades are critical."
                ),
                action_items=[
                    "Create database of pre-2006 buildings >4 floors",
                    "Prioritize hospitals, schools, and government buildings",
                    "Offer low-interest loans for voluntary retrofitting",
                    "Require re-inspection before building sale/lease",
                ],
                estimated_impact="Reduces potential earthquake casualties by up to 60%",
            ),
            ZoneRecommendation(
                zone_name="Uttara / Purbachal",
                priority="Medium",
                category="Density",
                title="Expand Planned Township Model to Peripheral Areas",
                description=(
                    "Uttara shows that planned grid development with adequate green space "
                    "and utilities can accommodate high density sustainably."
                ),
                action_items=[
                    "Replicate Uttara planning model in Purbachal New Town",
                    "Ensure 20% green space allocation from the start",
                    "Build schools, hospitals before residential occupation",
                    "Implement smart city infrastructure (fiber, IoT sensors)",
                ],
                estimated_impact="Relieves pressure on central Dhaka, improves livability",
            ),
        ]

    def to_dataframe(self, recommendations: List[ZoneRecommendation]) -> pd.DataFrame:
        return pd.DataFrame([{
            "Zone":      r.zone_name,
            "Priority":  r.priority,
            "Category":  r.category,
            "Title":     r.title,
            "Description": r.description,
            "Actions":   " | ".join(r.action_items),
            "Impact":    r.estimated_impact,
        } for r in recommendations])
