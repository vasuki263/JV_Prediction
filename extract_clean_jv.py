# extract_clean_jv.py

import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

MAX_IRRADIANCE_VARIATION = 45
MAX_IRRADIANCE_DIFFERENCE_FWD_BWD = 30
TRIM_START = 5
TRIM_END = 5

# Logging counters
LOG = {
    "files_scanned": 0,
    "forward_total": 0,
    "backward_total": 0,
    "forward_valid": 0,
    "backward_valid": 0,
    "forward_invalid": 0,
    "backward_invalid": 0
}

def extract_temperature(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
        m = re.search(r'(\d+\.?\d*)\s*°?C', first)
        return float(m.group(1)) if m else None

def load_raw(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        f.readline()
        data = np.genfromtxt(f, delimiter=';', skip_header=0)
    if data.ndim != 2 or data.shape[1] < 5:
        return None
    return data[:,2], data[:,3], data[:,4]

def compute_params(voltage, current):
    # Voc
    idx = np.argmin(np.abs(current))
    Voc = float(voltage[idx])

    # Isc
    idx2 = np.argmin(np.abs(voltage))
    Isc = float(current[idx2])

    # MPP
    power = voltage * current
    i = np.argmax(power)
    Vmpp = float(voltage[i])
    Impp = float(current[i])
    Pmpp = float(power[i])

    # FF
    FF = Pmpp / (Voc * Isc) if Voc*Isc != 0 else 0

    return Voc, Isc, Vmpp, Impp, Pmpp, FF


def extract_clean_jv(filepath):
    LOG["files_scanned"] += 1

    temp = extract_temperature(filepath)
    if temp is None:
        return []

    voltage, current, irr = load_raw(filepath)
    if voltage is None:
        return []

    sep = min(899, len(voltage)//2)
    irr_fwd, irr_bwd = irr[:sep], irr[sep:]

    # QC
    if np.max(irr_fwd)-np.min(irr_fwd) > MAX_IRRADIANCE_VARIATION:
        return []
    if np.max(irr_bwd)-np.min(irr_bwd) > MAX_IRRADIANCE_VARIATION:
        return []
    if abs(np.mean(irr_fwd)-np.mean(irr_bwd)) > MAX_IRRADIANCE_DIFFERENCE_FWD_BWD:
        return []

    # Trim
    v_fwd = voltage[:sep][TRIM_START:-TRIM_END]
    i_fwd = current[:sep][TRIM_START:-TRIM_END]
    v_bwd = voltage[sep:][TRIM_START:-TRIM_END]
    i_bwd = current[sep:][TRIM_START:-TRIM_END]

    # Compute parameters
    Voc_f, Isc_f, Vmpp_f, Impp_f, Pmpp_f, FF_f = compute_params(v_fwd, i_fwd)
    Voc_b, Isc_b, Vmpp_b, Impp_b, Pmpp_b, FF_b = compute_params(v_bwd, i_bwd)

    irr_avg = float(np.mean(irr))

    # -----------------------------
    # FILTER OUT INVALID SAMPLES
    # -----------------------------
    def valid_sample(Voc, Isc, Pmpp):
        if Voc <= 0: return False
        if Isc <= 0: return False
        if Pmpp <= 0: return False
        return True

    samples = []

    # Forward sweep
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
            "Voc": Voc_f,
            "Jsc": Isc_f,
            "Pmpp": Pmpp_f,
            "FF": FF_f
        })
    else:
        LOG["forward_invalid"] += 1

    # Backward sweep
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

    # -----------------------------
    # PRINT LOG SUMMARY
    # -----------------------------
    print("\n================ JV EXTRACTION SUMMARY ================")
    print(f"Files scanned:          {LOG['files_scanned']}")
    print(f"Forward sweeps total:   {LOG['forward_total']}")
    print(f"Backward sweeps total:  {LOG['backward_total']}")
    print(f"Forward valid:          {LOG['forward_valid']}")
    print(f"Backward valid:         {LOG['backward_valid']}")
    print(f"Forward invalid:        {LOG['forward_invalid']}")
    print(f"Backward invalid:       {LOG['backward_invalid']}")
    print("========================================================")

    # -----------------------------
    # HISTOGRAM OF VALIDITY
    # -----------------------------
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
