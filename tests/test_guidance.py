"""tests/test_guidance.py — Unit tests for guidance modules."""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.guidance.zoning_analyzer import ZoningAnalyzer
from src.guidance.risk_assessor import RiskAssessor
from src.guidance.recommendation import ModishCityRecommender


@pytest.fixture
def sample_buildings():
    """Create a sample GeoDataFrame with building data."""
    n = 100
    np.random.seed(42)
    return gpd.GeoDataFrame({
        "geometry": [Point(90.41 + np.random.uniform(-0.05, 0.05),
                           23.78 + np.random.uniform(-0.05, 0.05)) for _ in range(n)],
        "predicted_height": np.random.uniform(3, 100, n),
        "zone_name": np.random.choice(
            ["Gulshan", "Banani", "Old Dhaka", "Mirpur", "Motijheel"], n
        ),
    }, crs="EPSG:4326")


class TestZoningAnalyzer:
    def test_analyze_adds_columns(self, sample_buildings, tmp_path, monkeypatch):
        import yaml
        config = {
            "height_zoning": {
                "residential_low":    {"max_height_m": 12,  "far": 2.5, "ground_coverage": 0.60},
                "residential_medium": {"max_height_m": 27,  "far": 3.5, "ground_coverage": 0.55},
                "residential_high":   {"max_height_m": 60,  "far": 4.5, "ground_coverage": 0.50},
                "commercial_mixed":   {"max_height_m": 90,  "far": 6.0, "ground_coverage": 0.60},
                "special_commercial": {"max_height_m": 150, "far": 8.0, "ground_coverage": 0.55},
            }
        }
        cfg_path = tmp_path / "guidance_config.yaml"
        cfg_path.write_text(yaml.dump(config))
        analyzer = ZoningAnalyzer(config_path=str(cfg_path))
        result = analyzer.analyze_buildings(sample_buildings)

        assert "height_violation" in result.columns
        assert "zone_category" in result.columns
        assert "max_allowed_m" in result.columns

    def test_violation_logic(self, sample_buildings, tmp_path):
        import yaml
        config = {
            "height_zoning": {
                "residential_low":    {"max_height_m": 12,  "far": 2.5, "ground_coverage": 0.60},
                "residential_medium": {"max_height_m": 27,  "far": 3.5, "ground_coverage": 0.55},
                "residential_high":   {"max_height_m": 60,  "far": 4.5, "ground_coverage": 0.50},
                "commercial_mixed":   {"max_height_m": 90,  "far": 6.0, "ground_coverage": 0.60},
                "special_commercial": {"max_height_m": 150, "far": 8.0, "ground_coverage": 0.55},
            }
        }
        cfg_path = tmp_path / "guidance_config.yaml"
        cfg_path.write_text(yaml.dump(config))
        analyzer = ZoningAnalyzer(config_path=str(cfg_path))

        gdf = sample_buildings.copy()
        gdf["predicted_height"] = 200.0   # Force violation
        result = analyzer.analyze_buildings(gdf)
        assert result["height_violation"].all()


class TestRiskAssessor:
    def test_risk_columns_added(self, sample_buildings):
        assessor = RiskAssessor()
        result = assessor.assess_structural_risk(sample_buildings)
        for col in ["structural_overload_risk", "seismic_risk", "risk_score", "risk_level"]:
            assert col in result.columns

    def test_risk_score_range(self, sample_buildings):
        assessor = RiskAssessor()
        result = assessor.assess_structural_risk(sample_buildings)
        assert result["risk_score"].between(0, 3).all()


class TestRecommender:
    def test_generates_recommendations(self):
        zone_summary = pd.DataFrame({
            "zone_name":   ["Gulshan", "Old Dhaka"],
            "mean_height": [28.0, 5.0],
            "max_height":  [95.0, 22.0],
            "total_buildings": [1420, 3800],
            "violations": [170, 836],
            "violation_pct": [12.0, 22.0],
        })
        risk_matrix  = pd.DataFrame()
        buildings_gdf = gpd.GeoDataFrame({
            "geometry": [Point(90.41, 23.78)],
            "predicted_height": [20.0],
            "zone_name": ["Gulshan"],
        }, crs="EPSG:4326")

        recommender = ModishCityRecommender()
        recs = recommender.generate(zone_summary, risk_matrix, buildings_gdf)
        assert len(recs) > 0
        for r in recs:
            assert r.priority in ["Critical", "High", "Medium", "Low"]

    def test_to_dataframe(self):
        recommender = ModishCityRecommender()
        zone_summary = pd.DataFrame({
            "zone_name":   ["Gulshan"], "mean_height": [25.0],
            "max_height":  [90.0], "total_buildings": [500],
            "violations": [50], "violation_pct": [10.0],
        })
        recs = recommender.generate(zone_summary, pd.DataFrame(),
                                     gpd.GeoDataFrame({"geometry":[],"predicted_height":[],"zone_name":[]}))
        df = recommender.to_dataframe(recs)
        assert isinstance(df, pd.DataFrame)
        assert "Zone" in df.columns
        assert "Priority" in df.columns
