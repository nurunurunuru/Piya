# 🏙️ Building Height Analysis Using Deep Learning — Dhaka City & Modish City Guidance

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive deep learning pipeline for estimating **building heights in Dhaka City** from satellite imagery and providing **AI-driven urban planning guidance** for modern, sustainable city development.

---

## 📌 Project Objectives

1. **Height Estimation**: Predict per-building heights (meters) from Sentinel-2 multispectral imagery and auxiliary geospatial data using deep learning.
2. **Spatial Analysis**: Map height distributions across Dhaka's administrative zones (Gulshan, Banani, Mirpur, Old Dhaka, Motijheel, etc.).
3. **Modish City Guidance**: Provide AI-driven recommendations for modern urban planning — optimal FAR, setback rules, risk zones, green space allocation.

---

## 🗂️ Project Structure

```
dhaka-height-analysis/
├── README.md
├── requirements.txt
├── setup.py
├── Makefile
├── .env.example
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── data_config.yaml
│   └── guidance_config.yaml
├── data/
│   ├── raw/            # Satellite imagery, LiDAR, OSM
│   ├── processed/      # Patches, DSM, features
│   └── results/        # Predictions, maps, reports
├── notebooks/          # Jupyter exploration notebooks
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # DL architectures
│   ├── training/       # Training loop & callbacks
│   ├── evaluation/     # Metrics & visualization
│   ├── inference/      # Prediction & post-processing
│   └── guidance/       # Modish city planning module
├── pipeline/           # End-to-end pipeline stages
├── dashboard/          # Streamlit web dashboard
└── tests/              # Unit & integration tests
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-org/dhaka-height-analysis.git
cd dhaka-height-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (GEE, etc.)
```

### 3. Run the Full Pipeline

```bash
# Run all stages sequentially
python pipeline/run_pipeline.py --config configs/training_config.yaml

# Or run individual stages
python pipeline/stage_01_ingest.py
python pipeline/stage_02_preprocess.py
python pipeline/stage_03_train.py
python pipeline/stage_04_predict.py
python pipeline/stage_05_analyze.py
python pipeline/stage_06_report.py
```

### 4. Launch Backend API & Frontend

#### Step A: Run Backend API (FastAPI)
In terminal 1:
```bash
python backend.py
# Server running at http://localhost:5000
```

#### Step B: Run Web UI Frontend
In terminal 2:
```bash
python -m http.server 8000 --directory web_ui
# Open http://localhost:8000 in your browser
```

#### Alternative: Streamlit Dashboard
```bash
streamlit run dashboard/app.py
# Opens dashboard at http://localhost:8501
```

### 5. Using Docker

```bash
docker-compose up --build
```

---

## 🧠 Model Architecture

### Primary: U-Net with ResNet-50 Encoder
- Input: 12-channel Sentinel-2 image patches (256×256 px)
- Output: Single-channel height regression map
- Encoder: Pretrained ResNet-50 (ImageNet)
- Decoder: Transposed convolutions with skip connections

### Secondary: Swin Transformer
- Global context modeling for large-area height patterns
- Window attention: 8×8 patches
- 4-stage hierarchical feature extraction

### Ensemble
- Weighted average: 0.6 × U-Net + 0.4 × Swin Transformer

---

## 📊 Model Performance (Target Benchmarks)

| Metric | Target | Description |
|--------|--------|-------------|
| RMSE   | < 3.5m | Root Mean Square Error |
| MAE    | < 2.5m | Mean Absolute Error |
| R²     | > 0.85 | Coefficient of Determination |
| δ1     | > 0.80 | % predictions within 1.25× ground truth |

---

## 🏙️ Modish City Guidance Features

- **Zone Classification**: Automatically classifies areas by development density
- **FAR Analysis**: Floor Area Ratio compliance checking
- **Risk Assessment**: Seismic vulnerability, flood risk, structural safety
- **Green Space Planner**: Identifies zones for parks and open spaces
- **Height Zoning Advisor**: Recommends optimal building heights per zone
- **PDF/HTML Report Generator**: Automated planning reports

---

## 📍 Dhaka Zones Covered

| Zone | Area (km²) | Characteristics |
|------|-----------|----------------|
| Gulshan | 12.5 | High-rise commercial/residential |
| Banani | 8.2 | Mixed high-density |
| Dhanmondi | 11.8 | Medium-density residential |
| Motijheel | 6.4 | Central business district |
| Old Dhaka | 14.2 | Dense low-rise heritage |
| Mirpur | 18.6 | Residential, expanding |
| Uttara | 13.9 | Planned modern township |
| Mohakhali | 7.1 | Mixed commercial |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.0+, segmentation_models_pytorch |
| Geospatial | GDAL, Rasterio, GeoPandas, Shapely |
| Remote Sensing | Google Earth Engine, sentinelsat |
| Visualization | Folium, Kepler.gl, Plotly, Matplotlib |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Testing | Pytest |

---

## 📁 Data Sources

| Source | Type | Resolution | Access |
|--------|------|-----------|--------|
| Sentinel-2 L2A | Multispectral | 10m | GEE / Copernicus |
| SRTM/NASADEM | DEM | 30m | GEE |
| OpenStreetMap | Building footprints | Vector | OSMnx |
| Dhaka RAJUK GIS | Official zoning | Vector | RAJUK portal |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- RAJUK (Rajdhani Unnayan Kartripakkha) for urban data
- ESA Copernicus Programme for Sentinel-2 imagery
- OpenStreetMap contributors
- Bangladesh Space Research and Remote Sensing Organization (SPARRSO)
