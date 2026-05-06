from influxdb_client import InfluxDBClient
import pandas as pd
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
import os

# ---------------------------------------------------------
# LOAD .env FILE
# ---------------------------------------------------------
load_dotenv()

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET]):
    raise ValueError("Missing one or more required InfluxDB environment variables.")


class InfluxDBDataExporter:
    """
    Class for querying time-aggregated ParkData from InfluxDB
    and exporting results to pandas DataFrames and CSV files.
    """

    MEASUREMENT = "ParkData"

    def __init__(self,
                 url: str = INFLUX_URL,
                 token: str = INFLUX_TOKEN,
                 org: str = INFLUX_ORG,
                 bucket: str = INFLUX_BUCKET,
                 verify_ssl: bool = True):

        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.verify_ssl = verify_ssl
        self.client: Optional[InfluxDBClient] = None
        self._connect()

    # ---------------------------------------------------------
    # CONNECT
    # ---------------------------------------------------------
    def _connect(self) -> None:
        try:
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                verify_ssl=self.verify_ssl
            )
            print("\nConnected to InfluxDB")

            health = self.client.health()
            print(f"Health check: {health.status} — {health.message}")

        except Exception as e:
            print(f"Connection error: {e}")
            self.client = None
            raise

    # ---------------------------------------------------------
    # QUERY BUILDER
    # ---------------------------------------------------------
    def _build_query(self, module_names: List[str], start_date: str, end_date: str) -> str:
        name_filter = " or ".join([f'r.Name == "{name}"' for name in module_names])

        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: time(v: "{start_date}T00:00:00Z"), stop: time(v: "{end_date}T23:59:59Z"))
          |> filter(fn: (r) => r._measurement == "{self.MEASUREMENT}")
          |> filter(fn: (r) => {name_filter})
          |> aggregateWindow(every: 10m, fn: mean, createEmpty: false)
          |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
          |> keep(columns: ["_time", "Name", "I", "U", "P", "Temp", "AmbTemp", "AmbHmd", "Irr"])
        """
        return query

    # ---------------------------------------------------------
    # QUERY EXECUTION
    # ---------------------------------------------------------
    def query_data(self, module_names: List[str], start_date: str, end_date: str,
                   save_to_csv: bool = True, output_path: Optional[str] = None) -> pd.DataFrame:

        if self.client is None:
            raise ConnectionError("Not connected to InfluxDB.")

        if not module_names:
            raise ValueError("Module names list cannot be empty.")

        try:
            print(f"\nQuerying {len(module_names)} modules")
            print(f"Date range: {start_date} to {end_date}")
            print(f"Modules: {', '.join(module_names)}")

            query = self._build_query(module_names, start_date, end_date)
            df = self.client.query_api().query_data_frame(query)

            if isinstance(df, list):
                df = pd.concat(df, ignore_index=True)

            if df.empty:
                print("No data returned for the given range and module names.")
                return pd.DataFrame()

            df = df.sort_values("_time").reset_index(drop=True)

            if save_to_csv:
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"influx_export_{timestamp}.csv"

                df.to_csv(output_path, index=False)
                print(f"Data saved to: {output_path}")

            print(f"Retrieved {len(df)} rows")
            return df

        except Exception as e:
            print(f"Query error: {e}")
            raise

    # ---------------------------------------------------------
    # MODULE LISTS
    # ---------------------------------------------------------
    def get_si_modules(self) -> List[str]:
        return [
            "Atersa_1_1", "Atersa_2_1", "Atersa_3_1", "Atersa_4_1", "Atersa_5_1", "Atersa_6_1",
            "Atersa_1-1", "Atersa_2-1", "Atersa_3-1", "Atersa_4-1", "Atersa_5-1", "Atersa_6-1",
            "Sanyo_1_1", "Sanyo_2_1", "Sanyo_3_1", "Sanyo_4_1", "Sanyo_5_1",
            "Sanyo_1-1", "Sanyo_2-1", "Sanyo_3-1", "Sanyo_4-1", "Sanyo_5-1",
            "Solon_1_1", "Solon_1_2", "Solon_2_1", "Solon_2_2", "Solon_3_1", "Solon_3_2", "Solon_4_2",
            "Solon_1-1", "Solon_1-2", "Solon_2-1", "Solon_2-2", "Solon_3-1", "Solon_3-2", "Solon_4-2",
            "Sun_Power_1_1", "Sun_Power_2_1", "Sun_Power_3_1", "Sun_Power_4_1", "Sun_Power_5_1",
            "Sun_Power_1-1", "Sun_Power_2-1", "Sun_Power_3-1", "Sun_Power_4-1", "Sun_Power_5-1"
        ]

    def get_psc_modules(self) -> List[str]:
        return [
            "Perovskite_1", "Perovskite_1_1",
            "Perovskite_2", "Perovskite_1_2",
            "Perovskite_1_3",
            "Perovskite_1_4",
            "Perovskite_2_1",
            "Perovskite_2_2",
            "Perovskite_2_3",
            "Perovskite_3_1",
            "Perovskite_3_2",
            "Perovskite_3_3",
            "Perovskite_4_1",
            "Perovskite_4_2",
            "Perovskite_4_3",
        ]

    def get_all_modules(self) -> List[str]:
        return self.get_si_modules() + self.get_psc_modules()

    # ---------------------------------------------------------
    # CLOSE CONNECTION
    # ---------------------------------------------------------
    def close(self):
        if self.client is not None:
            self.client.close()
            print("InfluxDB connection closed")


# ---------------------------------------------------------
# EXAMPLE USAGE (now using .env)
# ---------------------------------------------------------
if __name__ == "__main__":
    exporter = InfluxDBDataExporter()

    psk_modules = exporter.get_psc_modules()
    df_psk = exporter.query_data(
        module_names=psk_modules,
        start_date="2026-01-01",
        end_date="2026-04-01",
        save_to_csv=True,
        output_path="perovskite_influx.csv"
    )

    si_modules = exporter.get_si_modules()
    df_si = exporter.query_data(
        module_names=si_modules,
        start_date="2026-01-01",
        end_date="2026-04-01",
        save_to_csv=True,
        output_path="silicon_influx.csv"
    )

    print(df_psk.head())
    print(df_si.head())