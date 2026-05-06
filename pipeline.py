"""
pipeline.py
Master entry script to run the full JV forecasting pipeline
for BOTH Perovskite and Silicon, with logging + timestamped runs
and automatic InfluxDB data export.
"""

import argparse
import os
import sys
import logging
from datetime import datetime

from get_influxdb_pv_data import InfluxDBDataExporter
from build_dataset import build_dataset
from train_lstm import train
from evaluate_lstm import evaluate


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
TECH_CONFIG = {
    "Silicon": {
        "data_dir": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Si_raw",
        "influx_csv": "silicon_influx.csv",
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "model_path": "models/jv_model_silicon.pth"
    },
    "Perovskite": {
        "data_dir": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Psk_raw",
        "influx_csv": "perovskite_influx.csv",
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "model_path": "models/jv_model_perovskite.pth"
    }
}


# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------
# STEP 0 — InfluxDB Export
# ---------------------------------------------------------
def run_influx_export(tech, logger):
    logger.info(f"=== STEP 0: Downloading latest InfluxDB data for {tech} ===")

    exporter = InfluxDBDataExporter()

    if tech == "Silicon":
        modules = exporter.get_si_modules()
        output_csv = "silicon_influx.csv"
    else:
        modules = exporter.get_psc_modules()
        output_csv = "perovskite_influx.csv"

    df = exporter.query_data(
        module_names=modules,
        start_date="2026-01-01",
        end_date="2026-04-01",
        save_to_csv=True,
        output_path=output_csv
    )

    logger.info(f"InfluxDB export complete for {tech}. Saved to {output_csv}")


# ---------------------------------------------------------
# STEP 1 — Build Dataset
# ---------------------------------------------------------
def run_build(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 1: Building dataset for {tech} ===")
    build_dataset(cfg["data_dir"], influx_csv=cfg["influx_csv"])
    logger.info(f"Dataset build complete for {tech}.\n")


# ---------------------------------------------------------
# STEP 2 — Train Model
# ---------------------------------------------------------
def run_train(tech, logger, run_dir):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 2: Training LSTM model for {tech} ===")
    train(cfg["data_dir"], influx_csv=cfg["influx_csv"],
          model_path=cfg["model_path"], run_dir=run_dir)
    logger.info(f"Training complete for {tech}.\n")


# ---------------------------------------------------------
# STEP 3 — Evaluate Model
# ---------------------------------------------------------
def run_evaluate(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 3: Evaluating model for {tech} ===")
    evaluate(
        cfg["data_dir"],
        influx_csv=cfg["influx_csv"],
        weather_csv=cfg["weather_csv"],
        model_path=cfg["model_path"]
    )
    logger.info(f"Evaluation complete for {tech}.\n")


# ---------------------------------------------------------
# FULL PIPELINE FOR ONE TECHNOLOGY
# ---------------------------------------------------------
def run_full_pipeline(tech):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"runs/{timestamp}_{tech}"
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, "pipeline.log"))

    logger.info(f"Starting full pipeline for {tech}")
    logger.info(f"Run directory: {run_dir}")

    run_influx_export(tech, logger)
    run_build(tech, logger)
    run_train(tech, logger, run_dir)
    run_evaluate(tech, logger)

    logger.info(f"Pipeline finished for {tech}")
    logger.info(f"Evaluation outputs saved in: outputs/{tech}/\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="JV Forecasting Pipeline Runner")

    parser.add_argument("--all", action="store_true", help="Run full pipeline for both technologies")
    parser.add_argument("--build", action="store_true", help="Build datasets only")
    parser.add_argument("--train", action="store_true", help="Train models only")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models only")
    parser.add_argument("--influx", action="store_true", help="Download latest InfluxDB data only")

    args = parser.parse_args()

    technologies = ["Silicon", "Perovskite"]

    # Default behavior: run full pipeline if no flags provided
    if not any([args.all, args.build, args.train, args.evaluate, args.influx]):
        args.all = True

    if args.all:
        for tech in technologies:
            run_full_pipeline(tech)
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for tech in technologies:
        run_dir = f"runs/{timestamp}_{tech}"
        os.makedirs(run_dir, exist_ok=True)
        logger = setup_logger(os.path.join(run_dir, "pipeline.log"))

        if args.influx:
            run_influx_export(tech, logger)

        if args.build:
            run_build(tech, logger)

        if args.train:
            run_train(tech, logger, run_dir)

        if args.evaluate:
            run_evaluate(tech, logger)


if __name__ == "__main__":
    main()
