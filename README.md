# Physics-Informed Machine Learning for FullJ–V Curve Prediction of Perovskite and Silicon Photovoltaic Module

This project provides an end-to-end pipeline for developing, training, and evaluating an **Autoencoder model for photovoltaic J–V (Current Density–Voltage) curve analysis**.

The pipeline supports both **Silicon** and **Perovskite** photovoltaic technologies and automates dataset generation, model training, evaluation, experiment logging, and artifact management.

## Research Objective

The objective of this research is to investigate the use of **Physics informed ML Autoencoders for learning compressed latent representations of photovoltaic J–V curves** under varying environmental conditions.

The model learns compact representations of measured J–V characteristics from both Silicon and Perovskite solar cells, capturing the underlying patterns and variations associated with changes in environmental conditions, particularly temperature and irradiance.

These learned latent representations provide a foundation for subsequent analysis and physics-informed modelling of photovoltaic performance.

## Supported Technologies

| Technology | Dataset / Panel |
|---|---|
| Silicon | `SunPower_5-1` |
| Perovskite | `Perovskite_5&6` |

## Project Workflow

The pipeline consists of three main stages:

### 1. Dataset Generation

Generates an Autoencoder-ready dataset by combining:

- Measured J–V curve data
- Weather information
- Technology-specific preprocessing

**Output:**

```text
data/
└── jv_autoencoder_dataset_*.npz
```

### 2. Autoencoder Training

Trains a neural Autoencoder using the generated dataset.

**Features:**

- Latent-space compression
- J–V curve reconstruction learning
- Automatic model artifact storage
- Run-specific experiment output directories
- Training and experiment logging

**Outputs:**

```text
models/
└── jv_autoencoder_*.pth

runs/
└── <timestamp>_<technology>/
```

### 3. Autoencoder Evaluation

Evaluates the trained Autoencoder and reconstructs photovoltaic J–V curves.

**Evaluation includes:**

- Reconstruction performance analysis
- Sample J–V curve visualizations
- Comparison of measured and reconstructed characteristics
- Predicted J–V curve analysis

## Configuration

The pipeline is configured through the `TECH_CONFIG` dictionary.

Each technology has its own configuration, specifying:

- Weather dataset location
- Raw weather data source
- Autoencoder dataset output path
- Autoencoder model output path

**Supported configurations:**

- `SunPower_5-1`
- `Perovskite_5&6`

## Running the Pipeline

### Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For Windows environments requiring the PyTorch-specific dependencies:

```bash
pip install -r requirements-torch_windows.txt
```

Optional environment variables can be stored in a `.env` file.

The project automatically loads environment variables using:

```python
load_dotenv()
```

### Train the Autoencoder

**For Silicon:**

```bash
python pipeline_runner.py --tech "SunPower_5-1" --train-ae
```

**For Perovskite:**

```bash
python pipeline_runner.py --tech "Perovskite_5&6" --train-ae
```

The training pipeline automatically:

1. Generates the Autoencoder dataset.
2. Trains the Autoencoder model.
3. Saves the trained model and associated artifacts.
4. Stores experiment logs in a run-specific directory.

### Evaluate the Autoencoder

**For Silicon:**

```bash
python eval_AE.py --tech "SunPower_5-1" --model-path "<model_path>" --dataset-path "<dataset_path>"
```

**For Perovskite:**

```bash
python eval_AE.py --tech "Perovskite_5&6" --model-path "<model_path>" --dataset-path "<dataset_path>"
```

The evaluation pipeline:

1. Loads the trained Autoencoder model.
2. Loads the corresponding evaluation dataset.
3. Reconstructs the J–V curves.
4. Generates evaluation results and visualizations.

## Output Artifacts

The pipeline generates the following artifacts:

### Datasets

```text
data/
└── *.npz
```

Contains the processed datasets used for Autoencoder training and evaluation.

### Trained Models

```text
models/
└── *.pth
```

Contains the saved PyTorch model checkpoints.

### Experiment Logs

```text
runs/
└── <timestamp>_<technology>/
    └── pipeline.log
```

Contains run-specific logs and experiment outputs.

## Technologies and Dependencies

The project uses:

- **Python** – Core programming language
- **PyTorch** – Autoencoder implementation and training
- **NumPy** – Numerical operations and dataset handling
- **Pandas** – Data processing
- **Matplotlib** – Visualization
- **python-dotenv** – Environment variable management

For the complete dependency list, refer to `requirements.txt` and `requirements-torch_windows.txt`.

---

*This repository forms part of research into data-driven and physics-informed modelling of photovoltaic J–V characteristics across different solar cell technologies and environmental operating conditions.*
