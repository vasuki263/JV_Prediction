import os
import io
import re
import numpy as np
import pandas as pd
import boto3
from datetime import datetime
from influxdb_client import InfluxDBClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

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
    "backward_invalid": 0,
}

# ---------------------------------------------------------
# Global clients + cache
# ---------------------------------------------------------
S3_CLIENT = boto3.client("s3")

INFLUX_CLIENT = InfluxDBClient(
    url=os.getenv("INFLUX_URL"),
    token=os.getenv("INFLUX_TOKEN"),
    org=os.getenv("INFLUX_ORG"),
)

CACHE_DIR = "cache_jv"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Parse timestamp from filename
# ---------------------------------------------------------
def parse_timestamp_from_filename(name):
    m = re.search(r"(\d{8})_(\d{6})", name)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return None


# ---------------------------------------------------------
# 2. Compute JV parameters
# ---------------------------------------------------------
def compute_params(voltage, current):
    idx_voc = np.argmin(np.abs(current))
    Voc = float(voltage[idx_voc])

    idx_isc = np.argmin(np.abs(voltage))
    Isc = float(current[idx_isc])

    power = voltage * current
    idx_mpp = np.argmax(power)

    Vmpp = float(voltage[idx_mpp])
    Impp = float(current[idx_mpp])
    Pmpp = float(power[idx_mpp])

    FF = Pmpp / (Voc * Isc) if Voc * Isc != 0 else 0
    return Voc, Isc, Vmpp, Impp, Pmpp, FF


# ---------------------------------------------------------
# 3. Parse a single JV file from bytes
# ---------------------------------------------------------
def extract_clean_jv_from_bytes(data, key):
    LOG["files_scanned"] += 1

    df = pd.read_csv(io.BytesIO(data))
    voltage = df["voltage"].astype(float).values
    current = df["current"].astype(float).values
    irr = df["irradiance"].astype(float).values
    temp = float(df["temperature"].iloc[0])

    timestamp = parse_timestamp_from_filename(key)
    if timestamp is None:
        return []

    # Forward/backward split
    sep = min(899, len(voltage) // 2)
    irr_fwd, irr_bwd = irr[:sep], irr[sep:]

    # Irradiance QC
    if (
        np.max(irr_fwd) - np.min(irr_fwd) > MAX_IRRADIANCE_VARIATION
        or np.max(irr_bwd) - np.min(irr_bwd) > MAX_IRRADIANCE_VARIATION
        or abs(np.mean(irr_fwd) - np.mean(irr_bwd)) > MAX_IRRADIANCE_DIFFERENCE_FWD_BWD
    ):
        return []

    # Trim sweeps
    v_fwd = voltage[:sep][TRIM_START:-TRIM_END]
    i_fwd = current[:sep][TRIM_START:-TRIM_END]
    v_bwd = voltage[sep:][TRIM_START:-TRIM_END]
    i_bwd = current[sep:][TRIM_START:-TRIM_END]

    # Compute parameters
    Voc_f, Isc_f, Vmpp_f, Impp_f, Pmpp_f, FF_f = compute_params(v_fwd, i_fwd)
    Voc_b, Isc_b, Vmpp_b, Impp_b, Pmpp_b, FF_b = compute_params(v_bwd, i_bwd)

    irr_avg = float(np.mean(irr))

    def valid(Voc, Isc, Pmpp):
        return Voc > 0 and Isc > 0 and Pmpp > 0

    samples = []

    # Forward sweep
    LOG["forward_total"] += 1
    if valid(Voc_f, Isc_f, Pmpp_f):
        LOG["forward_valid"] += 1
        samples.append({
            "file": key,
            "sweep": "forward",
            "temperature": temp,
            "irradiance": irr_avg,
            "voltage": v_fwd,
            "current": i_fwd,
            "timestamp": timestamp,
            "Voc": Voc_f,
            "Jsc": Isc_f,
            "Pmpp": Pmpp_f,
            "FF": FF_f,
        })
    else:
        LOG["forward_invalid"] += 1

    # Backward sweep
    LOG["backward_total"] += 1
    if valid(Voc_b, Isc_b, Pmpp_b):
        LOG["backward_valid"] += 1
        samples.append({
            "file": key,
            "sweep": "backward",
            "temperature": temp,
            "irradiance": irr_avg,
            "voltage": v_bwd,
            "current": i_bwd,
            "timestamp": timestamp,
            "Voc": Voc_b,
            "Jsc": Isc_b,
            "Pmpp": Pmpp_b,
            "FF": FF_b,
        })
    else:
        LOG["backward_invalid"] += 1

    return samples


# ---------------------------------------------------------
# 4. Query InfluxDB for S3 keys
# ---------------------------------------------------------
def get_s3_keys_from_influx(tech):
    query = f"""
    from(bucket: "test_jv_upload")
      |> range(start: -365d)
      |> filter(fn: (r) => r["_measurement"] == "jv_measurements")
      |> filter(fn: (r) => r["panel_name"] == "{tech}")
      |> filter(fn: (r) => r["_field"] == "s3_bucket" or r["_field"] == "s3_object_key")
    """

    tables = INFLUX_CLIENT.query_api().query(query)
    pairs = {}

    for table in tables:
        for row in table.records:
            t = row["_time"]
            pairs.setdefault(t, {"bucket": None, "key": None})

            if row["_field"] == "s3_bucket":
                pairs[t]["bucket"] = row["_value"]
            elif row["_field"] == "s3_object_key":
                pairs[t]["key"] = row["_value"]

    keys = [(d["bucket"], d["key"]) for d in pairs.values() if d["bucket"] and d["key"]]

    print("DEBUG: extracted key-count:", len(keys))
    return keys


# ---------------------------------------------------------
# 5. Cached S3 download
# ---------------------------------------------------------
def download_one(bucket, key):
    local_path = os.path.join(CACHE_DIR, key.replace("/", "_"))

    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return key, f.read()

    obj = S3_CLIENT.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    with open(local_path, "wb") as f:
        f.write(data)

    return key, data


# ---------------------------------------------------------
# 6. Extract all JV samples (parallel)
# ---------------------------------------------------------
def extract_all_from_influx(tech):
    keys = get_s3_keys_from_influx(tech)
    if not keys:
        print("No S3 keys found for tech:", tech)
        return []

    print(f"Starting parallel download of {len(keys)} JV files...")

    samples = []
    futures = []

    with ThreadPoolExecutor(max_workers=20) as ex:
        for bucket, key in keys:
            futures.append(ex.submit(download_one, bucket, key))

        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading JV files ({tech})"):
            try:
                key, data = f.result()
                extracted = extract_clean_jv_from_bytes(data, key)
                if extracted:
                    samples.extend(extracted)
            except Exception as e:
                print(f"[WARN] Failed to process {key}: {e}")

    print("Total JV samples extracted:", len(samples))
    print("LOG:", LOG)
    return samples
