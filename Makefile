# ============================================================
# Makefile — Dhaka Building Height Analysis Pipeline
# ============================================================

.PHONY: help setup install test lint clean pipeline dashboard docker backend web_ui start

PYTHON=python
PIP=pip
PYTEST=pytest

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Launch App ----
start:  ## Start both Backend API and Web UI Frontend concurrently
	$(PYTHON) run_app.py

# ---- Environment ----
setup:  ## Create virtualenv and install dependencies
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

install:  ## Install package in dev mode
	$(PIP) install -e .

# ---- Pipeline Stages ----
ingest:  ## Stage 1: Data ingestion
	$(PYTHON) pipeline/stage_01_ingest.py

preprocess:  ## Stage 2: Data preprocessing
	$(PYTHON) pipeline/stage_02_preprocess.py

train:  ## Stage 3: Model training
	$(PYTHON) pipeline/stage_03_train.py --config configs/training_config.yaml

predict:  ## Stage 4: Run predictions
	$(PYTHON) pipeline/stage_04_predict.py

analyze:  ## Stage 5: Spatial analysis
	$(PYTHON) pipeline/stage_05_analyze.py

report:  ## Stage 6: Generate reports
	$(PYTHON) pipeline/stage_06_report.py

pipeline:  ## Run full pipeline end-to-end
	$(PYTHON) pipeline/run_pipeline.py

# ---- Backend API ----
backend:  ## Start FastAPI backend server (Port 5000)
	$(PYTHON) backend.py

# ---- Web UI Frontend ----
web_ui:  ## Start Web UI frontend server (Port 8000)
	$(PYTHON) -m http.server 8000 --directory web_ui

# ---- Dashboard ----
dashboard:  ## Launch Streamlit dashboard (Port 8501)
	streamlit run dashboard/app.py --server.port 8501

# ---- MLflow ----
mlflow:  ## Start MLflow tracking server
	mlflow ui --host 0.0.0.0 --port 5000

# ---- Testing ----
test:  ## Run all tests
	$(PYTEST) tests/ -v --cov=src --cov-report=html

test-fast:  ## Run tests without slow integration tests
	$(PYTEST) tests/ -v -m "not slow"

# ---- Code Quality ----
lint:  ## Run linting checks
	flake8 src/ pipeline/ dashboard/ --max-line-length=120
	black src/ pipeline/ dashboard/ --check

format:  ## Auto-format code
	black src/ pipeline/ dashboard/
	isort src/ pipeline/ dashboard/

# ---- Docker ----
docker-build:  ## Build Docker image
	docker build -t dhaka-height-analysis:latest -f docker/Dockerfile .

docker-up:  ## Start Docker services
	docker-compose -f docker/docker-compose.yml up --build -d

docker-down:  ## Stop Docker services
	docker-compose -f docker/docker-compose.yml down

# ---- Cleanup ----
clean:  ## Remove generated artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ dist/ build/

clean-data:  ## Remove processed data (keep raw)
	rm -rf data/processed/* data/results/*
