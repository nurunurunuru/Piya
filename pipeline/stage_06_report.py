"""
pipeline/stage_06_report.py
Stage 6: Generate Modish City guidance report (HTML + PDF).
"""
import logging
import pandas as pd
logger = logging.getLogger(__name__)


def run(config: dict):
    import os
    from src.guidance.zoning_analyzer import ZoningAnalyzer
    from src.guidance.risk_assessor import RiskAssessor
    from src.guidance.recommendation import ModishCityRecommender
    from src.guidance.report_generator import ReportGenerator
    import geopandas as gpd

    osm_path  = "data/raw/osm/osm_buildings_dhaka.gpkg"
    out_dir   = "data/results/reports"
    os.makedirs(out_dir, exist_ok=True)

    # Load buildings
    if os.path.exists(osm_path):
        gdf = gpd.read_file(osm_path)
        gdf["predicted_height"] = 10.0  # Placeholder
        gdf["zone_name"] = "Unknown"
    else:
        logger.warning("OSM data not found. Using sample data for report.")
        import numpy as np
        zones = ["Gulshan", "Banani", "Dhanmondi", "Motijheel",
                 "Old Dhaka", "Mirpur", "Uttara", "Mohakhali"]
        n = 800
        gdf = gpd.GeoDataFrame({
            "predicted_height": np.random.uniform(3, 70, n),
            "zone_name": np.random.choice(zones, n),
            "height_violation": np.random.choice([True, False], n, p=[0.15, 0.85]),
            "max_allowed_m": np.random.choice([12, 27, 60, 90], n),
        })

    # Zoning analysis
    analyzer     = ZoningAnalyzer()
    gdf          = analyzer.analyze_buildings(gdf)
    zone_summary = analyzer.compute_zone_summary(gdf)

    # Risk assessment
    risk_assessor = RiskAssessor()
    gdf           = risk_assessor.assess_structural_risk(gdf)
    risk_matrix   = risk_assessor.generate_risk_matrix(gdf)

    # Recommendations
    recommender     = ModishCityRecommender()
    recommendations = recommender.generate(zone_summary, risk_matrix, gdf)

    rec_df = recommender.to_dataframe(recommendations)
    rec_df.to_csv(f"{out_dir}/recommendations.csv", index=False)

    # Generate report
    reporter = ReportGenerator()
    html_path = reporter.generate_html(
        zone_summary=zone_summary,
        risk_matrix=risk_matrix,
        recommendations=recommendations,
        output_path=f"{out_dir}/modish_city_report.html",
        total_buildings=len(gdf),
    )
    reporter.generate_pdf(html_path, f"{out_dir}/modish_city_report.pdf")

    logger.info(f"Report saved → {out_dir}/modish_city_report.html")
    logger.info(f"Generated {len(recommendations)} recommendations.")
