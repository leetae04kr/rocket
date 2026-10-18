import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd
import numpy as np

from pipeline.config import (
    DATA_DIR,
    DEFAULT_MAX_Q_LIMIT_KPA,
    RAW_LL2_DIR
)

logger = logging.getLogger("GCATCollector")

GCAT_DIR = DATA_DIR / "raw_gcat"
GCAT_DIR.mkdir(parents=True, exist_ok=True)

LAUNCHLOG_URL = "https://planet4589.org/space/gcat/tsv/derived/launchlog.tsv"
SITES_URL = "https://planet4589.org/space/gcat/tsv/tables/sites.tsv"


class GCATDataCollector:
    """
    Jonathan McDowell's General Catalog of Artificial Space Objects (GCAT) Launch Log collector.
    Contains the definitive, comprehensive historical orbital launch record (1957-Present, 7400+ launches).
    Bypasses API rate limits entirely and downloads full historical data instantly.
    """

    def __init__(self, cache_dir: Path = GCAT_DIR):
        self.cache_dir = cache_dir
        self.launchlog_path = self.cache_dir / "launchlog.tsv"
        self.sites_path = self.cache_dir / "sites.tsv"

    def ensure_data_downloaded(self):
        """Downloads latest TSV database dumps if not present locally."""
        if not self.sites_path.exists():
            logger.info("Downloading GCAT sites coordinate database...")
            r = requests.get(SITES_URL, timeout=60)
            r.raise_for_status()
            self.sites_path.write_text(r.text, encoding="utf-8")

        if not self.launchlog_path.exists():
            logger.info("Downloading GCAT full historical launch database (1957-Present)...")
            r = requests.get(LAUNCHLOG_URL, timeout=120)
            r.raise_for_status()
            self.launchlog_path.write_text(r.text, encoding="utf-8")

    def load_site_coordinates(self) -> Dict[str, Tuple[float, float, str]]:
        """Parses sites.tsv into site_code -> (latitude, longitude, site_name) dictionary."""
        site_coords = {}
        with open(self.sites_path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline()
            headers = [c.strip().lstrip("#") for c in header_line.split("\t")]
            # skip second comment line
            f.readline()
            df_sites = pd.read_csv(f, sep="\t", names=headers, on_bad_lines="skip")

        df_sites["Site"] = df_sites["Site"].astype(str).str.strip()
        for _, row in df_sites.iterrows():
            try:
                lat = float(str(row.get("Latitude", "")).strip())
                lon = float(str(row.get("Longitude", "")).strip())
                name = str(row.get("Name", row.get("ShortName", "Unknown Pad"))).strip()
                site_coords[row["Site"]] = (lat, lon, name)
            except (ValueError, TypeError):
                continue

        logger.info(f"Loaded {len(site_coords)} launch site coordinates.")
        return site_coords

    def collect_all_launches(
        self,
        max_records: Optional[int] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Parses all historical launches into normalized schema compatible with the pipeline.
        Target:
          OS (Orbital Success), DS (Deep Space Success) -> 1
          OF (Orbital Failure), OP (Partial Failure) -> 0
        """
        self.ensure_data_downloaded()
        site_map = self.load_site_coordinates()

        logger.info("Parsing GCAT launch records...")
        with open(self.launchlog_path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline()
            headers = [c.strip().lstrip("#") for c in header_line.split("\t")]
            f.readline()  # skip update date line
            df_raw = pd.read_csv(f, sep="\t", names=headers, low_memory=False, on_bad_lines="skip")

        # Each row can represent a payload/piece. De-duplicate by unique launch tag
        df_unique = df_raw.drop_duplicates(subset=["Launch_Tag"]).copy()
        logger.info(f"Loaded {len(df_unique)} unique historical launch missions.")

        records: List[Dict[str, Any]] = []

        for _, row in df_unique.iterrows():
            code = str(row.get("Launch_Code", "")).strip().upper()
            # Map Success / Failure
            if code.startswith("OS") or code.startswith("DS") or code == "S":
                target = 1
            elif code.startswith("OF") or code.startswith("OP") or code.startswith("F"):
                target = 0
            else:
                # Skip suborbital test, uncataloged, or ambiguous codes
                continue

            # Parse date
            raw_date = str(row.get("Launch_Date", "")).strip()
            dt = self._parse_launch_date(raw_date)
            if dt is None or pd.isna(dt):
                continue

            year = dt.year
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue

            # Site coordinates
            site_code = str(row.get("Launch_Site", "")).strip()
            coords = site_map.get(site_code)
            if not coords:
                continue
            lat, lon, pad_name = coords

            launch_id = str(row.get("Launch_Tag", "")).strip()
            rocket_name = str(row.get("LV_Type", "Unknown Rocket")).strip()
            mission_name = str(row.get("Name", rocket_name)).strip()

            rocket_type = self._infer_rocket_type(rocket_name)
            max_q_kpa = self._estimate_max_q(rocket_name)
            total_mass_kg, thrust_kn = self._estimate_specs(rocket_name)

            records.append({
                "launch_id": launch_id,
                "launch_name": f"{rocket_name} | {mission_name}",
                "launch_time_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pad_name": f"{site_code} - {pad_name}",
                "latitude": lat,
                "longitude": lon,
                "rocket_id": abs(hash(rocket_name)) % 10000,
                "rocket_name": rocket_name,
                "rocket_type": rocket_type,
                "total_mass_kg": total_mass_kg,
                "liftoff_thrust_kn": thrust_kn,
                "max_q_limit_kpa": max_q_kpa,
                "payload_mass_kg": None,
                "launch_status": target
            })

            if max_records and len(records) >= max_records:
                break

        df = pd.DataFrame(records)
        logger.info(f"Successfully processed {len(df)} launches from GCAT repository.")

        if save_intermediate and not df.empty:
            cache_file = RAW_LL2_DIR / "ll2_launches_extracted.parquet"
            df.to_parquet(cache_file, index=False)
            logger.info(f"Saved normalized GCAT dataset to {cache_file}")

        return df

    def _parse_launch_date(self, s: str) -> Optional[pd.Timestamp]:
        """Parses dates like '1957 Oct  4 1928:34', '2024 Feb 8 0633', or '2020 Jun 30'."""
        try:
            s_clean = re.sub(r"\s+", " ", str(s)).strip()
            # Case 1: YYYY Mon DD HHMM:SS
            m1 = re.search(r"(\d{4})\s+([A-Za-z]{3})\s+(\d{1,2})\s+(\d{2})(\d{2}):(\d{2})", s_clean)
            if m1:
                y, m, d, hh, mm, ss = m1.groups()
                return pd.to_datetime(f"{y}-{m}-{int(d):02d} {hh}:{mm}:{ss}", utc=True)

            # Case 2: YYYY Mon DD HHMM (without seconds)
            m2 = re.search(r"(\d{4})\s+([A-Za-z]{3})\s+(\d{1,2})\s+(\d{2})(\d{2})", s_clean)
            if m2:
                y, m, d, hh, mm = m2.groups()
                return pd.to_datetime(f"{y}-{m}-{int(d):02d} {hh}:{mm}:00", utc=True)

            # Case 3: YYYY Mon DD (date only)
            m3 = re.search(r"(\d{4})\s+([A-Za-z]{3})\s+(\d{1,2})", s_clean)
            if m3:
                y, m, d = m3.groups()
                return pd.to_datetime(f"{y}-{m}-{int(d):02d} 00:00:00", utc=True)

            dt = pd.to_datetime(s_clean, utc=True, errors="coerce")
            return None if pd.isna(dt) else dt
        except Exception:
            return None

    def _infer_rocket_type(self, name: str) -> str:
        name_lower = name.lower()
        if any(h in name_lower for h in ["heavy", "saturn v", "sls", "starship", "energya"]):
            return "Heavy"
        if any(s in name_lower for s in ["electron", "scout", "pegasus", "black brant", "minotaur i"]):
            return "Small"
        return "Medium"

    def _estimate_max_q(self, rocket_name: str) -> float:
        name_lower = rocket_name.lower()
        for family, limit in DEFAULT_MAX_Q_LIMIT_KPA.items():
            if family in name_lower:
                return limit
        return DEFAULT_MAX_Q_LIMIT_KPA["default"]

    def _estimate_specs(self, rocket_name: str) -> Tuple[Optional[float], Optional[float]]:
        """Provides baseline mass (kg) and thrust (kN) estimates for major launch vehicle families."""
        nl = rocket_name.lower()
        if "falcon 9" in nl:
            return 549054.0, 7607.0
        if "falcon heavy" in nl:
            return 1420788.0, 22819.0
        if "soyuz" in nl or "r-7" in nl:
            return 313000.0, 4150.0
        if "proton" in nl:
            return 705000.0, 10530.0
        if "atlas v" in nl:
            return 334500.0, 3827.0
        if "delta iv" in nl:
            return 249500.0, 3140.0
        if "ariane 5" in nl:
            return 777000.0, 11400.0
        if "long march" in nl or "cz-" in nl:
            return 450000.0, 6000.0
        if "electron" in nl:
            return 13000.0, 162.0
        if "space shuttle" in nl:
            return 2040000.0, 30160.0
        if "saturn v" in nl:
            return 2970000.0, 35100.0
        return None, None
