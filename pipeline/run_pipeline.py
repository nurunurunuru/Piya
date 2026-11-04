"""
pipeline/run_pipeline.py
Master pipeline orchestrator — runs all 6 stages sequentially.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

load_dotenv()
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")

STAGES = [
    ("Stage 1: Data Ingestion",      "pipeline.stage_01_ingest"),
    ("Stage 2: Preprocessing",       "pipeline.stage_02_preprocess"),
    ("Stage 3: Model Training",      "pipeline.stage_03_train"),
    ("Stage 4: Prediction",          "pipeline.stage_04_predict"),
    ("Stage 5: Spatial Analysis",    "pipeline.stage_05_analyze"),
    ("Stage 6: Report Generation",   "pipeline.stage_06_report"),
]


def run_stage(module_name: str, config: dict) -> bool:
    """Dynamically import and run a pipeline stage."""
    import importlib
    try:
        module = importlib.import_module(module_name)
        module.run(config)
        return True
    except Exception as e:
        logger.error(f"Stage failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Dhaka Building Height Analysis Pipeline")
    parser.add_argument("--config", default="configs/training_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--stages", nargs="+", type=int,
                        help="Run specific stages only (e.g. --stages 1 2 3)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (use existing raw data)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open("configs/data_config.yaml") as f:
        config["data"] = yaml.safe_load(f)
    with open("configs/model_config.yaml") as f:
        config["model_cfg"] = yaml.safe_load(f)
    with open("configs/guidance_config.yaml") as f:
        config["guidance"] = yaml.safe_load(f)

    config["skip_download"] = args.skip_download

    # Create output directories
    for d in ["logs", "data/raw", "data/processed", "data/results",
              "models/checkpoints"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]🏙️ Dhaka Building Height Analysis Pipeline[/bold cyan]\n"
        "[dim]Deep Learning + Modish City Guidance[/dim]",
        border_style="cyan",
    ))

    selected_stages = args.stages or list(range(1, len(STAGES) + 1))
    start_total = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline...", total=len(selected_stages))

        for stage_num in selected_stages:
            stage_name, module = STAGES[stage_num - 1]
            progress.update(task, description=f"[yellow]{stage_name}[/yellow]")
            console.print(f"\n[bold]{'─'*60}[/bold]")
            console.print(f"[bold green]▶ {stage_name}[/bold green]")

            t0 = time.time()
            success = run_stage(module, config)
            elapsed = time.time() - t0

            if success:
                console.print(f"[green]✓ Completed in {elapsed:.1f}s[/green]")
            else:
                console.print(f"[red]✗ Stage failed![/red]")
                console.print("[yellow]Check logs/pipeline.log for details.[/yellow]")
                break

            progress.advance(task)

    total_time = time.time() - start_total
    console.print(Panel(
        f"[bold green]✅ Pipeline Complete![/bold green]\n"
        f"Total time: {total_time/60:.1f} minutes\n\n"
        f"Results: [cyan]data/results/[/cyan]\n"
        f"Report:  [cyan]data/results/reports/modish_city_report.html[/cyan]\n"
        f"Dashboard: [cyan]streamlit run dashboard/app.py[/cyan]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
