import io
import os
import zipfile
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz
import requests


class DWDDownloader:
    """
    Downloads DWD weather data (10-min and hourly), cleans, merges, validates,
    interpolates, converts units, and saves a final CSV.
    """

    BASE_URL = (
        "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate"
    )

    ERROR_VALUES = {
        "temp_C": -999,
        "humidity": -999,
        "cloud_cover": -1,
        "precip_mm": -999,
        "precip_indicator": -999,
        "sunshine_duration": -999,
        "global_radiation": -999,
    }

    CONTINUOUS_COLS = ["temp_C", "humidity", "precip_mm", "global_radiation"]
    DISCRETE_COLS = ["cloud_cover", "precip_indicator", "sunshine_duration"]

    DROP_METADATA_COLS = [
        "STATIONS_ID", "Stations_ID", "Stations_id",
        "Stationsname", "Stations_Name",
        "Stationshoehe", "Geogr.Breite", "Geogr.Laenge",
        "Von_Datum", "Bis_Datum", "von_datum", "bis_datum",
        "eor", "Parameter", "Beschreibung",
        "Gesamt_Fehlwerte", "Anzahl_Fehlwerte",
    ]

    def __init__(
        self,
        station_id: str = "04931",
        solar_station_id: str = "04928",
        output_path: Optional[str] = None,
        tmp_dir: str = "tmp_dwd",
    ):
        self.station_id = station_id
        self.solar_station_id = solar_station_id
        self.output_path = output_path
        self.tmp_dir = tmp_dir

        self.german_tz = pytz.timezone("Europe/Berlin")
        os.makedirs(self.tmp_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Download + extract helpers
    # ---------------------------------------------------------
    def _download_and_extract(self, category: str, filename: str) -> List[str]:
        url = f"{self.BASE_URL}/{category}/recent/{filename}"
        extracted_files = []

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                for name in zf.namelist():
                    if name.endswith(".txt"):
                        out_path = os.path.join(self.tmp_dir, os.path.basename(name))
                        with open(out_path, "wb") as f:
                            f.write(zf.read(name))
                        extracted_files.append(out_path)

        except Exception as exc:
            print(f"Failed DWD data download/extract: {url}\n{exc}")

        return extracted_files

    def _read_and_combine_txt(self, files: List[str]) -> pd.DataFrame:
        dfs = []
        for path in files:
            try:
                df = pd.read_csv(path, sep=";", comment="#", na_values="-999")
                df.columns = df.columns.str.strip()
                dfs.append(df)
            except Exception as exc:
                print(f"Skipping {path}: {exc}")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # ---------------------------------------------------------
    # Timestamp cleaning
    # ---------------------------------------------------------
    def _clean_timestamps(self, df: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
        if df.empty or "timestamp" not in df:
            return df

        fmt = "%Y%m%d%H%M" if freq == "10min" else "%Y%m%d%H"

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], format=fmt, errors="coerce")
        df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp")

        df = (
            df.set_index("timestamp")
            .tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
            .tz_convert("UTC")
            .reset_index()
        )

        return df.drop_duplicates("timestamp")

    # ---------------------------------------------------------
    # Individual dataset loaders
    # ---------------------------------------------------------
    def get_temperature_humidity(self) -> pd.DataFrame:
        files = self._download_and_extract(
            "10_minutes/air_temperature",
            f"10minutenwerte_TU_{self.station_id}_akt.zip",
        )
        df = self._read_and_combine_txt(files)
        return df.rename(columns={"MESS_DATUM": "timestamp", "TT_10": "temp_C", "RF_10": "humidity"})

    def get_cloudiness(self) -> pd.DataFrame:
        files = self._download_and_extract(
            "hourly/cloudiness",
            f"stundenwerte_N_{self.station_id}_akt.zip",
        )
        df = self._read_and_combine_txt(files)
        return df.rename(columns={"MESS_DATUM": "timestamp", "V_N": "cloud_cover"})

    def get_precipitation(self) -> pd.DataFrame:
        files = self._download_and_extract(
            "10_minutes/precipitation",
            f"10minutenwerte_nieder_{self.station_id}_akt.zip",
        )
        df = self._read_and_combine_txt(files)
        return df.rename(columns={"MESS_DATUM": "timestamp", "RWS_10": "precip_mm", "RWS_IND_10": "precip_indicator"})

    def get_solar(self) -> pd.DataFrame:
        files = self._download_and_extract(
            "10_minutes/solar",
            f"10minutenwerte_SOLAR_{self.solar_station_id}_akt.zip",
        )
        df = self._read_and_combine_txt(files)
        return df.rename(columns={"MESS_DATUM": "timestamp", "GS_10": "global_radiation", "SD_10": "sunshine_duration"})

    # ---------------------------------------------------------
    # Merge + clean
    # ---------------------------------------------------------
    def _merge_dataframes(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        cleaned = []
        for df in dfs:
            df = df.drop(columns=[c for c in self.DROP_METADATA_COLS if c in df.columns], errors="ignore")
            cleaned.append(df)

        merged = cleaned[0]
        for df in cleaned[1:]:
            merged = pd.merge(merged, df, on="timestamp", how="outer")

        return merged.drop_duplicates("timestamp").sort_values("timestamp")

    def _validate_and_interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Replace error values with NaN
        for col, err in self.ERROR_VALUES.items():
            if col in df:
                df[col] = df[col].replace(err, np.nan)

        # Continuous → interpolate
        for col in self.CONTINUOUS_COLS:
            if col in df:
                df[col] = df[col].interpolate(limit_direction="both")

        # Discrete → forward/backward fill
        for col in self.DISCRETE_COLS:
            if col in df:
                df[col] = df[col].ffill().bfill()

        # Fill remaining NaNs
        df = df.fillna(0)
        return df

    @staticmethod
    def _convert_solar_units(df: pd.DataFrame) -> pd.DataFrame:
        if "global_radiation" in df:
            df["global_radiation"] = df["global_radiation"] * 10000 / 600
        return df

    # ---------------------------------------------------------
    # Main pipeline
    # ---------------------------------------------------------
    def run(self) -> pd.DataFrame:
        print("\nDownloading DWD weather data")

        temp = self._clean_timestamps(self.get_temperature_humidity())
        cloud = self._clean_timestamps(self.get_cloudiness(), freq="hour")
        precip = self._clean_timestamps(self.get_precipitation())
        solar = self._clean_timestamps(self.get_solar())

        # Resample cloud to 10-min
        if not cloud.empty and "cloud_cover" in cloud.columns:
            cloud = (
                cloud.drop_duplicates("timestamp")
                .assign(timestamp=lambda df: pd.to_datetime(df["timestamp"], errors="coerce"))
                .dropna(subset=["timestamp"])
                .sort_values("timestamp")
                .set_index("timestamp")
                .resample("10min")
                .ffill()
                .reset_index()
            )

        merged = self._merge_dataframes([temp, precip, solar, cloud])
        cleaned = self._validate_and_interpolate(merged)
        cleaned = self._convert_solar_units(cleaned)

        output = self.output_path or "weather_data_10min_cleaned.csv"
        cleaned.to_csv(output, index=False)

        print(f"Saved cleaned data -> {output}")
        return cleaned


if __name__ == "__main__":
    downloader = DWDDownloader(
        station_id="04931",
        solar_station_id="04928",
        output_path="weather_data_10min_cleaned.csv",
    )
    downloader.run()