import argparse
import os
import sys
import logging
from datetime import datetime

from dotenv import load_dotenv

# from jv_vs_irr import analyze_irradiance_AE
# from jv_vs_temp import analyze_temperature_AE

load_dotenv()

from jv_weather_pipeline import build_jv_weather_pipeline
from build_jv_ae_dataset import build_jv_autoencoder_dataset
from train_autoencoder import train as train_ae
from evaluate_autoencoder import evaluate_diode_model



# ---------------------------------------------------------
# Configuration (UPDATED FOR LSTM)
# ---------------------------------------------------------
TECH_CONFIG = {
    "SunPower_5-1": {
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "raw_weather": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv",

        # LSTM paths (NEW)
        "lstm_dataset": "data/jv_dataset_SunPower_5-1.npz",
        "lstm_model_path": "models/jv_lstm_silicon.pth",

        # AE paths (unchanged)
        "ae_dataset": "data/jv_autoencoder_dataset_silicon_temp_heatmap.npz",
        "ae_model_path": "models/jv_autoencoder_silicon_test.pth",
    },

    "Perovskite_5&6": {
        "weather_csv": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_jv_cleaned.csv",
        "raw_weather": "C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/weather_data_10min_cleaned.csv",

        # LSTM paths (NEW)
        "lstm_dataset": "data/jv_dataset_Perovskite_5&6.npz",
        "lstm_model_path": "models/jv_lstm_perovskite.pth",

        # AE paths (unchanged)
        "ae_dataset": "data/jv_autoencoder_dataset_perovskite_iv3_test.npz",
        "ae_model_path": "models/jv_autoencoder_perovskite_iv3_test.pth",
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

    preds, df = evaluate_diode_model(
        model_path=cfg["ae_model_path"],
        dataset_path=cfg["ae_dataset"],
        tech=cfg["tech"],
        num_curves=10
    )

    logger.info("AE evaluation complete")



# ---------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------
def run_pipeline(cfg, args, logger, run_dir):

    # AE dataset build (needed for AE train/eval)
    if args.train_ae or args.all:
        step_ae_dataset(cfg, logger)
        step_train_ae(cfg, logger, run_dir)

    # AE evaluation
    if args.eval_ae or args.all:
        # step_ae_dataset(cfg, logger)
        step_eval_ae(cfg, logger)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="JV Forecasting Pipeline Runner")

    parser.add_argument("--tech", type=str, help="SunPower_5-1 or Perovskite_5&6")

    # LSTM controls
    parser.add_argument("--train-lstm", action="store_true", help="Train LSTM only")
    parser.add_argument("--eval-lstm", action="store_true", help="Evaluate LSTM only")

    # AE controls
    parser.add_argument("--train-ae", action="store_true", help="Train Autoencoder only")
    parser.add_argument("--eval-ae", action="store_true", help="Evaluate Autoencoder only")

    # Temperature analysis
    parser.add_argument("--temp", action="store_true", help="Run temperature analysis")

    # Irradiance analysis
    parser.add_argument("--irr", action="store_true", help="Run irradiance analysis")

    # Temp vs JV curve analysis
    parser.add_argument("--temp_jv", action="store_true", help="Run temp vs JV curves analysis")

    # Irr vs JV curve analysis
    parser.add_argument("--irr_jv", action="store_true", help="Run irradiance vs JV curves analysis")

    # Run everything
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    if not args.tech:
        print("Please specify --tech SunPower_5-1 or --tech Perovskite_5&6")
        return

    cfg = TECH_CONFIG[args.tech].copy()
    cfg["tech"] = args.tech

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"runs/{timestamp}_{args.tech}"
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(os.path.join(run_dir, "pipeline.log"))
    logger.info(f"Starting pipeline for {args.tech}")
    logger.info(f"Run directory: {run_dir}")

    run_pipeline(cfg, args, logger, run_dir)

    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
