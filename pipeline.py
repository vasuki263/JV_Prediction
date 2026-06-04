import argparse
import os
import sys
import logging
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from jv_weather_pipeline import build_jv_weather_pipeline
from build_dataset import build_dataset
from train_lstm import train as train_lstm
from evaluate_lstm import evaluate as eval_lstm
from build_jv_ae_dataset import build_jv_autoencoder_dataset
from train_autoencoder import train as train_ae
from evaluate_autoencoder import evaluate_autoencoder as eval_ae


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TECH_CONFIG = {
    "SunPower_5-1": {
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "raw_weather": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv",
        "model_path": "models/jv_model_silicon.pth",
        "ae_dataset": "data/jv_autoencoder_dataset_silicon.npz",
        "ae_model_path": "models/jv_autoencoder_silicon.pth",
    },
    "Perovskite_5&6": {
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "raw_weather": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv",
        "model_path": "models/jv_model_perovskite.pth",
        "ae_dataset": "data/jv_autoencoder_dataset_perovskite.npz",
        "ae_model_path": "models/jv_autoencoder_perovskite.pth",
    }
}


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
def setup_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------
def step_weather(cfg, logger):
    logger.info("=== STEP 0: Updating weather data ===")
    build_jv_weather_pipeline(cfg["raw_weather"], cfg["weather_csv"])
    logger.info(f"Weather updated --> {cfg['weather_csv']}")


def step_dataset(cfg, logger):
    logger.info("=== STEP 1: Building dataset ===")
    dataset = build_dataset(
        tech=cfg["tech"],
        clean_weather_path=cfg["weather_csv"]
    )
    if dataset is None:
        logger.error("Dataset build returned None")
    else:
        logger.info("Dataset build complete")
    return dataset


def step_train_lstm(cfg, logger, run_dir, dataset):
    logger.info("=== STEP 2: Training LSTM ===")
    train_lstm(dataset, cfg["tech"], model_path=cfg["model_path"], run_dir=run_dir)
    logger.info(f"LSTM model saved --> {cfg['model_path']}")


def step_eval_lstm(cfg, logger):
    logger.info("=== STEP 3: Evaluating LSTM ===")
    eval_lstm(model_path=cfg["model_path"], tech=cfg["tech"], weather_csv=cfg["weather_csv"])
    logger.info("LSTM evaluation complete")


def step_ae_dataset(cfg, logger):
    logger.info("=== STEP AE-1: Building AE dataset ===")
    build_jv_autoencoder_dataset(
        tech=cfg["tech"],
        weather_csv=cfg["weather_csv"],
        out_file=cfg["ae_dataset"]
    )
    logger.info(f"AE dataset saved --> {cfg['ae_dataset']}")


def step_train_ae(cfg, logger, run_dir):
    logger.info("=== STEP AE-2: Training Autoencoder ===")
    train_ae(
        dataset_path=cfg["ae_dataset"],
        model_path=cfg["ae_model_path"],
        out_dir=os.path.join(run_dir, "ae"),
        latent_dim=32
    )
    logger.info(f"AE model saved --> {cfg['ae_model_path']}")


def step_eval_ae(cfg, logger):
    logger.info("=== STEP AE-3: Evaluating Autoencoder ===")
    eval_ae(
        model_path=cfg["ae_model_path"],
        dataset_path=cfg["ae_dataset"],
        tech=cfg["tech"]
    )
    logger.info("AE evaluation complete")


# ---------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------
def run_full_pipeline(tech, run_ae=False):
    cfg = TECH_CONFIG[tech].copy()
    cfg["tech"] = tech

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"runs/{timestamp}_{tech}"
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, "pipeline.log"))
    logger.info(f"Starting pipeline for {tech}")
    logger.info(f"Run directory: {run_dir}")

    step_weather(cfg, logger)

    dataset = step_dataset(cfg, logger)
    if dataset is None:
        return

    step_train_lstm(cfg, logger, run_dir, dataset)
    step_eval_lstm(cfg, logger)

    if run_ae:
        step_ae_dataset(cfg, logger)
        step_train_ae(cfg, logger, run_dir)
        step_eval_ae(cfg, logger)

    logger.info(f"Pipeline finished for {tech}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="JV Forecasting Pipeline Runner")
    parser.add_argument("--tech", type=str, help="SunPower_5-1 or Perovskite_5&6")
    parser.add_argument("--all", action="store_true", help="Run for all technologies")
    parser.add_argument("--ae", action="store_true", help="Include Autoencoder steps")
    args = parser.parse_args()

    if args.all:
        for tech in TECH_CONFIG:
            run_full_pipeline(tech, run_ae=args.ae)
        return

    if not args.tech:
        print("Please specify --tech SunPower_5-1 or --tech Perovskite_5&6")
        return

    run_full_pipeline(args.tech, run_ae=args.ae)


if __name__ == "__main__":
    main()
