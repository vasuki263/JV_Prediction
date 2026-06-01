# pipeline.py — FINAL VERSION (InfluxDB-based JV extraction)

import argparse
import os
import sys
import logging
from datetime import datetime

from jv_weather_pipeline import build_jv_weather_pipeline
from build_dataset import build_dataset
from train_lstm import train
from evaluate_lstm import evaluate
from build_jv_ae_dataset_from_influx import build_jv_autoencoder_dataset
from train_autoencoder import train as train_ae
from evaluate_autoencoder import evaluate_autoencoder as evaluate_ae
from dotenv import load_dotenv
load_dotenv()

TECH_CONFIG = {
    "SunPower_5-1": {
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "raw_weather": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv",
        "model_path": "models/jv_model_silicon.pth",
        "ae_dataset": "data/jv_autoencoder_dataset_silicon.npz",
        "ae_model_path": "models/jv_autoencoder_silicon.pth"
    },
    "Perovskite_5&6": {
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "raw_weather": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv",
        "model_path": "models/jv_model_perovskite.pth",
        "ae_dataset": "data/jv_autoencoder_dataset_perovskite.npz",
        "ae_model_path": "models/jv_autoencoder_perovskite.pth"
    }
}

# ---------------------------------------------------------
# Logging
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
# Weather update
# ---------------------------------------------------------
def run_weather_update(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 0: Updating weather data for {tech} ===")

    df = build_jv_weather_pipeline(cfg["raw_weather"], cfg["weather_csv"])
    logger.info(f"Weather update complete → {cfg['weather_csv']}")

# ---------------------------------------------------------
# Dataset build
# ---------------------------------------------------------
def run_build(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 1: Building dataset for {tech} ===")

    result = build_dataset(tech, clean_weather_path=cfg["weather_csv"])
    if result is None:
        logger.error(f"No dataset built for {tech}")
        return None

    logger.info(f"Dataset build complete for {tech}")
    return result

def run_build_ae(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP AE-1: Building Autoencoder dataset for {tech} ===")

    build_jv_autoencoder_dataset(
        tech=tech,
        weather_csv=cfg["weather_csv"],
        out_file=cfg["ae_dataset"]
    )

    logger.info(f"AE dataset built → {cfg['ae_dataset']}")

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
def run_train(tech, logger, run_dir, dataset):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 2: Training LSTM model for {tech} ===")

    train(dataset, tech, model_path=cfg["model_path"], run_dir=run_dir)
    logger.info(f"Training complete → {cfg['model_path']}")

def run_train_ae(tech, logger, run_dir):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP AE-2: Training Autoencoder for {tech} ===")

    train_ae(
    dataset_path=cfg["ae_dataset"],
    model_path=cfg["ae_model_path"],
    out_dir=os.path.join(run_dir, "ae"),
    latent_dim=32
    )

    logger.info(f"AE training complete → {cfg['ae_model_path']}")

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
def run_evaluate(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP 3: Evaluating model for {tech} ===")

    evaluate(model_path=cfg["model_path"], tech=tech,
        weather_csv=cfg["weather_csv"])
    logger.info(f"Evaluation complete for {tech}")

def run_evaluate_ae(tech, logger):
    cfg = TECH_CONFIG[tech]
    logger.info(f"=== STEP AE-3: Evaluating Autoencoder for {tech} ===")

    evaluate_ae(
        model_path=cfg["ae_model_path"],
        dataset_path=cfg["ae_dataset"],
        tech=tech
    )

    logger.info(f"AE evaluation complete for {tech}")

# ---------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------
def run_full_pipeline(tech, run_ae=False):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"runs/{timestamp}_{tech}"
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, "pipeline.log"))
    logger.info(f"Starting full pipeline for {tech}")
    logger.info(f"Run directory: {run_dir}")

    run_weather_update(tech, logger)
    dataset = run_build(tech, logger)
    if dataset is None:
        return

    run_train(tech, logger, run_dir, dataset)
    run_evaluate(tech, logger)
    if run_ae:
        run_build_ae(tech, logger)
        run_train_ae(tech, logger, run_dir)
        run_evaluate_ae(tech, logger)

    logger.info(f"Pipeline finished for {tech}")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="JV Forecasting Pipeline Runner")

    parser.add_argument("--tech", type=str, default=None,
                        help="Technology: SunPower_5-1 or Perovskite_5&6")
    parser.add_argument("--all", action="store_true",
                        help="Run full pipeline for both technologies")
    parser.add_argument("--ae", action="store_true",
                        help="Run Autoencoder pipeline steps")

    args = parser.parse_args()

    if args.all:
        for tech in TECH_CONFIG.keys():
            run_full_pipeline(tech)
        return

    if args.tech is None:
        print("Please specify --tech Silicon or --tech Perovskite_5&6")
        return

    run_full_pipeline(args.tech, run_ae=args.ae)

if __name__ == "__main__":
    main()
