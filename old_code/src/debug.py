# debug_voc_rawplot.py

import numpy as np
import matplotlib.pyplot as plt
from preprocess_jv import load_jv_data

# CHANGE THIS to one of your actual SEPIV files
file = r"C:/Users/Lenovo/Desktop/Master Thesis/jv_forecasting/data/Psk_raw/2025-07-25/SEPIV3/20250725_070149_SEPIV3.txt"

print("Loading file:", file)
data = load_jv_data(file)

if data is None:
    print("ERROR: Could not load JV file. Check path or QC filters.")
    exit()

voltage = data["voltage_fwd"]
current = data["current_fwd"]

print("Voltage range:", np.min(voltage), "to", np.max(voltage))
print("Current range:", np.min(current), "to", np.max(current))

plt.figure(figsize=(8,6))
plt.plot(voltage, current, marker="o")
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("Raw JV Curve (No Voc Calculation)")
plt.grid(True)
plt.show()

