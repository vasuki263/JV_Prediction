# extract_clean_jv.py

import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
from datetime import datetime

MAX_IRRADIANCE_VARIATION = 45
MAX_IRRADIANCE_DIFFERENCE_FWD_BWD = 30
TRIM_START = 5
TRIM_END = 5

LOG = {
    "files_scanned": 0,
    "forward_total": 0,
    "backward_total": 0,
    "forward_valid": 0,
    "backward_valid": 0,
    "forward_invalid": 0,
    "backward_invalid": 0
}

def parse_timestamp_from_filename(name):
    m = re.search(r'(\d{8})_(\d{6})', name)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return None

def extract_temperature(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
        m = re.search(r'([\d.]+)\s*°?C', first)
        return float(m.group(1)) if m else None

def load_raw(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        f.readline()
        try:
            data = np.genfromtxt(
                f, delimiter=';', usecols=(2, 3, 4), invalid_raise=False
            )
        except Exception:
            return None, None, None

    if data is None or data.ndim != 2 or data.shape[1] < 3:
        return None, None, None

    return data[:, 0], data[:, 1], data[:, 2]

def compute_params(voltage, current):
    idx = np.argmin(np.abs(current))
    Voc = float(voltage[idx])
    idx2 = np.argmin(np.abs(voltage))
    Isc = float(current[idx2])
    power = voltage * current
    i = np.argmax(power)
    Vmpp = float(voltage[i])
    Impp = float(current[i])
    Pmpp = float(power[i])
    FF = Pmpp / (Voc * Isc) if Voc * Isc != 0 else 0
    return Voc, Isc, Vmpp, Impp, Pmpp, FF

def extract_clean_jv(filepath):
    LOG["files_scanned"] += 1

    temp = extract_temperature(filepath)
    if temp is None:
        return []

    voltage, current, irr = load_raw(filepath)
    if voltage is None:
        return []

    timestamp = parse_timestamp_from_filename(filepath.name)
    if timestamp is None:
        return []

    sep = min(899, len(voltage)//2)
    irr_fwd, irr_bwd = irr[:sep], irr[sep:]

    if np.max(irr_fwd)-np.min(irr_fwd) > MAX_IRRADIANCE_VARIATION:
        return []
    if np.max(irr_bwd)-np.min(irr_bwd) > MAX_IRRADIANCE_VARIATION:
        return []
    if abs(np.mean(irr_fwd)-np.mean(irr_bwd)) > MAX_IRRADIANCE_DIFFERENCE_FWD_BWD:
        return []

    v_fwd = voltage[:sep][TRIM_START:-TRIM_END]
    i_fwd = current[:sep][TRIM_START:-TRIM_END]
    v_bwd = voltage[sep:][TRIM_START:-TRIM_END]
    i_bwd = current[sep:][TRIM_START:-TRIM_END]

    Voc_f, Isc_f, Vmpp_f, Impp_f, Pmpp_f, FF_f = compute_params(v_fwd, i_fwd)
    Voc_b, Isc_b, Vmpp_b, Impp_b, Pmpp_b, FF_b = compute_params(v_bwd, i_bwd)

    irr_avg = float(np.mean(irr))

    def valid_sample(Voc, Isc, Pmpp):
        return Voc > 0 and Isc > 0 and Pmpp > 0

    samples = []

    LOG["forward_total"] += 1
    if valid_sample(Voc_f, Isc_f, Pmpp_f):
        LOG["forward_valid"] += 1
        samples.append({
            "file": filepath.name,
            "sweep": "forward",
            "temperature": temp,
            "irradiance": irr_avg,
            "voltage": v_fwd,
            "current": i_fwd,
            "timestamp": timestamp,
            "Voc": Voc_f,
            "Jsc": Isc_f,
            "Pmpp": Pmpp_f,
            "FF": FF_f
        })
    else:
        LOG["forward_invalid"] += 1

    LOG["backward_total"] += 1
    if valid_sample(Voc_b, Isc_b, Pmpp_b):
        LOG["backward_valid"] += 1
        samples.append({
            "file": filepath.name,
            "sweep": "backward",
            "temperature": temp,
            "irradiance": irr_avg,
            "voltage": v_bwd,
            "current": i_bwd,
            "timestamp": timestamp,
            "Voc": Voc_b,
            "Jsc": Isc_b,
            "Pmpp": Pmpp_b,
            "FF": FF_b
        })
    else:
        LOG["backward_invalid"] += 1

    return samples

def extract_all(directory):
    directory = Path(directory)
    files = [f for f in directory.rglob("*.txt") if "SEPIV" in f.name.upper()]
    samples = []

    for f in files:
        samples.extend(extract_clean_jv(f))

    print("\n================ JV EXTRACTION SUMMARY ================")
    for k, v in LOG.items():
        print(f"{k}: {v}")
    print("========================================================")

    valid = LOG["forward_valid"] + LOG["backward_valid"]
    invalid = LOG["forward_invalid"] + LOG["backward_invalid"]

    plt.figure(figsize=(6,5))
    plt.bar(["Valid", "Invalid"], [valid, invalid], color=["green","red"])
    plt.title("JV Sweep Validity After QC + Filtering")
    plt.ylabel("Number of Sweeps")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig("jv_validity_histogram.png", dpi=150)
    plt.close()

    print("\n📊 Histogram saved as: jv_validity_histogram.png\n")

    return samples
