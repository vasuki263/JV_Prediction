import pandas as pd
import numpy as np
import os

def load_raw_weather(csv_path):
    print(f"--- MANUAL LOAD START: {csv_path} ---")
    
    # 1. Use the most basic engine to prevent C-level crashes
    try:
        # We only load the columns we need to save memory immediately
        target_cols = ['timestamp', 'temp_C', 'humidity', 'cloud_cover', 'precip_mm', 'global_radiation']
        
        df = pd.read_csv(
            csv_path, 
            engine='python', 
            on_bad_lines='skip',
            usecols=target_cols
        )
    except Exception as e:
        print(f"CRITICAL ERROR during read_csv: {e}")
        # If it fails, try loading WITHOUT usecols just in case names are slightly different
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip', nrows=100)
        print("Columns actually found in file:", df.columns.tolist())
        raise e

    # 2. Basic Cleaning
    df.columns = df.columns.str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # 3. Timezone handling (Match UTC as used in DWD data)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

    df = df.set_index('timestamp').sort_index()
    
    print(f"--- MANUAL LOAD SUCCESS: {df.shape} rows ---")
    return df

def build_jv_weather_pipeline(raw_path, clean_path):
    """Simplified pipeline to avoid recursion crashes"""
    if os.path.exists(clean_path):
        print(f"Using existing cleaned weather: {clean_path}")
        df = pd.read_csv(clean_path, index_col=0, parse_dates=True)
        # Ensure index is UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        return df
    
    df_raw = load_raw_weather(raw_path)
    # Basic interpolation for missing 10-min gaps
    df_clean = df_raw.resample('10min').mean().interpolate(method='linear')
    df_clean.to_csv(clean_path)
    return df_clean