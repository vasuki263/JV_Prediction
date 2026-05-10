"""
pipeline.py — FINAL VERSION (WEATHER‑AWARE, NO INFLUX)
"""

import argparse
import os
import sys
import logging
from datetime import datetime

from jv_weather_pipeline import build_jv_weather_pipeline
from build_dataset import build_dataset
from train_lstm import train
from evaluate_lstm import evaluate


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
TECH_CONFIG = {
    "Silicon": {
        "data_dir": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Si_raw",
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "raw_weather": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv",
        "model_path": "models/jv_model_silicon.pth"
    },
    "Perovskite": {
        "data_dir": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Psk_raw",
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "raw_weather": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv",
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
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------
# STEP 0 — Weather Update
# ---------------------------------------------------------
def run_weather_update(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 0: Updating weather data for {tech} ===")

    df = build_jv_weather_pipeline(cfg["raw_weather"], cfg["weather_csv"])

    logger.info(f"Weather update complete. Cleaned weather saved to: {cfg['weather_csv']}")


# ---------------------------------------------------------
# STEP 1 — Build Dataset
# ---------------------------------------------------------
def run_build(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 1: Building dataset for {tech} ===")
    build_dataset(cfg["data_dir"], clean_weather_path=cfg["weather_csv"])
    logger.info(f"Dataset build complete for {tech}.\n")


# ---------------------------------------------------------
# STEP 2 — Train Model
# ---------------------------------------------------------
def run_train(tech, logger, run_dir):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 2: Training LSTM model for {tech} ===")
    train(cfg["data_dir"], cfg["weather_csv"], model_path=cfg["model_path"], run_dir=run_dir)
    logger.info(f"Training complete for {tech}.\n")


# ---------------------------------------------------------
# STEP 3 — Evaluate Model
# ---------------------------------------------------------
def run_evaluate(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 3: Evaluating model for {tech} ===")
    evaluate(cfg["data_dir"], cfg["weather_csv"], model_path=cfg["model_path"])
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

    # Full pipeline sequence
    run_weather_update(tech, logger)
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
    parser.add_argument("--weather", action="store_true", help="Update weather data only")
    parser.add_argument("--build", action="store_true", help="Build datasets only")
    parser.add_argument("--train", action="store_true", help="Train models only")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models only")

    args = parser.parse_args()

    technologies = ["Silicon", "Perovskite"]

    # Default behavior: run full pipeline if no flags provided
    if not any([args.all, args.weather, args.build, args.train, args.evaluate]):
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

        if args.weather:
            run_weather_update(tech, logger)

        if args.build:
            run_build(tech, logger)

        if args.train:
            run_train(tech, logger, run_dir)

        if args.evaluate:
            run_evaluate(tech, logger)


if __name__ == "__main__":
    main()
