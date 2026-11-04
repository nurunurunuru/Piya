"""
pipeline/stage_05_analyze.py
Stage 5: Spatial analysis — zone statistics, height distribution,
interactive maps, and 3D visualization.
"""
import logging
import numpy as np
logger = logging.getLogger(__name__)


def run(config: dict):
    import geopandas as gpd
    from src.data.preprocessor import load_raster
    from src.evaluation.visualizer import (
        plot_height_map, plot_error_map, plot_zone_metrics,
        create_interactive_map, plot_3d_height_surface,
    )
    from src.guidance.zoning_analyzer import ZoningAnalyzer

    pred_path = "data/results/predictions/height_map_final.tif"
    osm_path  = "data/raw/osm/osm_buildings_dhaka.gpkg"
    out_dir   = "data/results/maps"
    import os; os.makedirs(out_dir, exist_ok=True)

    # Load predictions
    height_arr, meta = load_raster(pred_path)

    # --- Static height map ---
    plot_height_map(
        height_arr.squeeze(),
        title="Predicted Building Heights — Dhaka City (2022–2023)",
        output_path=f"{out_dir}/height_map_static.png",
    )

    # --- 3D surface ---
    plot_3d_height_surface(
        height_arr.squeeze(),
        output_path=f"{out_dir}/height_map_3d.html",
    )

    # --- Load and enrich buildings GeoDataFrame ---
    if os.path.exists(osm_path):
        gdf = gpd.read_file(osm_path)
        gdf["predicted_height"] = np.random.uniform(3, 50, len(gdf))  # Placeholder until join
        gdf["zone_name"] = "Unknown"  # Assign via spatial join with zone polygons

        # Zoning analysis
        analyzer = ZoningAnalyzer()
        gdf = analyzer.analyze_buildings(gdf)
        zone_summary = analyzer.compute_zone_summary(gdf)
        zone_summary.to_csv(f"{out_dir}/zone_summary.csv", index=False)
        logger.info(f"Zone summary saved → {out_dir}/zone_summary.csv")

        # Interactive map
        create_interactive_map(
            pred_path, gdf,
            output_path=f"{out_dir}/interactive_height_map.html",
        )
    else:
        logger.warning(f"OSM buildings not found: {osm_path}")

    logger.info(f"All maps saved to {out_dir}/")
