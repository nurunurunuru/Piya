"""
src/evaluation/visualizer.py
Visualization tools: height maps, error maps, zone-wise bar charts, 3D renders.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static Height Map
# ---------------------------------------------------------------------------

def plot_height_map(
    height_array: np.ndarray,
    title: str = "Predicted Building Heights — Dhaka",
    output_path: Optional[str] = None,
    vmax: float = 80.0,
    cmap: str = "plasma",
) -> plt.Figure:
    """Plot a 2D height raster with colorbar."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(height_array.squeeze(), cmap=cmap, vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Building Height (m)", fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Height map saved → {output_path}")
    return fig


def plot_error_map(
    pred: np.ndarray,
    target: np.ndarray,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot absolute error map."""
    error = np.abs(pred.squeeze() - target.squeeze())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(target.squeeze(), cmap="plasma", vmin=0, vmax=60)
    axes[0].set_title("Ground Truth (nDSM)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(pred.squeeze(), cmap="plasma", vmin=0, vmax=60)
    axes[1].set_title("Predicted Height", fontweight="bold")
    axes[1].axis("off")

    im = axes[2].imshow(error, cmap="hot_r", vmin=0, vmax=15)
    plt.colorbar(im, ax=axes[2], label="Absolute Error (m)")
    axes[2].set_title("Absolute Error Map", fontweight="bold")
    axes[2].axis("off")

    plt.suptitle("Building Height Prediction — Dhaka City", fontsize=14, y=1.01)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Zone-wise Bar Chart
# ---------------------------------------------------------------------------

def plot_zone_metrics(
    zone_metrics: Dict[str, Dict],
    output_path: Optional[str] = None,
) -> go.Figure:
    """Interactive Plotly bar chart of per-zone RMSE and R²."""
    zones = list(zone_metrics.keys())
    rmse  = [zone_metrics[z].get("rmse", 0) for z in zones]
    r2    = [zone_metrics[z].get("r2",   0) for z in zones]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="RMSE (m)", x=zones, y=rmse,
                         marker_color="crimson", yaxis="y"))
    fig.add_trace(go.Bar(name="R²",       x=zones, y=r2,
                         marker_color="steelblue", yaxis="y2"))

    fig.update_layout(
        title="Per-Zone Height Prediction Performance — Dhaka",
        xaxis_title="Administrative Zone",
        yaxis=dict(title="RMSE (m)", side="left"),
        yaxis2=dict(title="R²", overlaying="y", side="right", range=[0, 1]),
        barmode="group",
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99),
    )

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Zone metrics chart saved → {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Interactive Folium Map
# ---------------------------------------------------------------------------

def create_interactive_map(
    height_geotiff: str,
    buildings_gdf: gpd.GeoDataFrame,
    output_path: str,
    center: Tuple[float, float] = (23.8103, 90.4125),
) -> folium.Map:
    """
    Create an interactive Folium map with:
    - Building footprints colored by predicted height
    - Heatmap layer of height density
    """
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB dark_matter")

    # Color scale for heights
    colormap = folium.LinearColormap(
        colors=["#003366", "#0066cc", "#ffcc00", "#ff6600", "#cc0000"],
        vmin=0, vmax=80,
        caption="Building Height (m)",
    )
    colormap.add_to(m)

    # Add building footprints
    if "predicted_height" in buildings_gdf.columns:
        for _, row in buildings_gdf.iterrows():
            h = row.get("predicted_height", 0) or 0
            color = colormap(min(h, 80))
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda f, c=color: {
                    "fillColor": c,
                    "color": "white",
                    "weight": 0.3,
                    "fillOpacity": 0.75,
                },
                tooltip=f"Height: {h:.1f}m",
            ).add_to(m)

    # Add heatmap layer
    if "predicted_height" in buildings_gdf.columns:
        centroids = buildings_gdf.geometry.centroid
        heat_data = [
            [pt.y, pt.x, row["predicted_height"]]
            for pt, (_, row) in zip(centroids, buildings_gdf.iterrows())
            if row.get("predicted_height", 0) > 0
        ]
        HeatMap(heat_data, name="Height Heatmap", radius=12).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    logger.info(f"Interactive map saved → {output_path}")
    return m


# ---------------------------------------------------------------------------
# 3D Height Surface (Plotly)
# ---------------------------------------------------------------------------

def plot_3d_height_surface(
    height_array: np.ndarray,
    downsample: int = 4,
    output_path: Optional[str] = None,
    title: str = "3D Building Height Map — Dhaka City",
) -> go.Figure:
    """Create a 3D surface plot from a height raster."""
    h = height_array.squeeze()[::downsample, ::downsample]

    fig = go.Figure(data=[go.Surface(
        z=h,
        colorscale="plasma",
        colorbar=dict(title="Height (m)"),
        lighting=dict(ambient=0.4, diffuse=0.6, roughness=0.4, specular=0.5),
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="East →",
            yaxis_title="North →",
            zaxis_title="Height (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        template="plotly_dark",
    )

    if output_path:
        fig.write_html(output_path)
        logger.info(f"3D surface saved → {output_path}")
    return fig
