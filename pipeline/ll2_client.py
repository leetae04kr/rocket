import time
import logging
from typing import Generator, Dict, Any, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

from pipeline.config import (
    LL2_BASE_URL,
    LL2_RATE_LIMIT_SLEEP_SEC,
    LL2_MAX_RETRIES,
    LL2_TIMEOUT_SEC,
    LL2_PAGE_LIMIT,
    LL2_SUCCESS_STATUS_IDS,
    LL2_FAILURE_STATUS_IDS,
    DEFAULT_MAX_Q_LIMIT_KPA,
    RAW_LL2_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LL2Collector")


class LL2DataCollector:
    """
    Launch Library 2 (LL2) API mass data extraction client.
    Handles rate-limiting, exponential backoff, pagination, and raw payload normalization.
    """

    def __init__(self, base_url: str = LL2_BASE_URL, sleep_sec: float = LL2_RATE_LIMIT_SLEEP_SEC):
        self.base_url = base_url
        self.sleep_sec = sleep_sec
        self.session = self._create_resilient_session()

    def _create_resilient_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=LL2_MAX_RETRIES,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": "RocketAtmosphereDataPipeline/1.0 (ECMWF-LL2 Data Engineering Research)"
        })
        return session

    def fetch_launches_page(
        self,
        offset: int = 0,
        limit: int = LL2_PAGE_LIMIT,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """Fetch a single page with backoff, net__gte/lte filtering, and 429 handling."""
        params = {
            "mode": "detailed",
            "limit": limit,
            "offset": offset,
            "ordering": "net"  # Chronological order
        }
        if start_year:
            params["net__gte"] = f"{start_year}-01-01T00:00:00Z"
        if end_year:
            params["net__lte"] = f"{end_year}-12-31T23:59:59Z"
        
        attempt = 0
        while attempt < LL2_MAX_RETRIES:
            try:
                response = self.session.get(self.base_url, params=params, timeout=LL2_TIMEOUT_SEC)
                
                if response.status_code == 429:
                    try:
                        detail = response.json().get("detail", "")
                    except Exception:
                        detail = response.text[:100]
                    
                    retry_after = int(response.headers.get("Retry-After", 10))
                    
                    # If throttled for too long on production, failover to developer endpoint immediately
                    if "ll.thespacedevs.com" in self.base_url:
                        logger.warning(f"Production LL2 rate limit hit ({detail}). Instant failover to developer endpoint (lldev.thespacedevs.com)...")
                        self.base_url = "https://lldev.thespacedevs.com/2.2.0/launch/previous/"
                        time.sleep(1.0)
                        continue

                    if retry_after > 30:
                        logger.warning(f"Rate limited for {retry_after}s. Capping sleep to 15s...")
                        time.sleep(15.0)
                    else:
                        time.sleep(retry_after)
                    attempt += 1
                    continue
                
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                attempt += 1
                backoff = (2 ** attempt) * 2
                logger.warning(f"Request failed: {e}. Retrying in {backoff}s (Attempt {attempt}/{LL2_MAX_RETRIES})")
                time.sleep(backoff)

        raise RuntimeError(f"Failed to fetch LL2 data at offset {offset} after {LL2_MAX_RETRIES} attempts.")

    def collect_all_launches(
        self,
        max_records: Optional[int] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Iterates over all pagination pages until end of records or max_records is reached.
        Extracts and filters records suitable for ML model training.
        """
        records: List[Dict[str, Any]] = []
        offset = 0
        total_fetched = 0
        total_count = None

        logger.info(f"Starting LL2 mass extraction from {self.base_url}...")

        while True:
            logger.info(f"Fetching offset={offset}, limit={LL2_PAGE_LIMIT} (Collected: {len(records)})")
            try:
                data = self.fetch_launches_page(
                    offset=offset,
                    limit=LL2_PAGE_LIMIT,
                    start_year=start_year,
                    end_year=end_year
                )
            except RuntimeError as err:
                if "ll.thespacedevs.com" in self.base_url:
                    logger.warning("Production LL2 rate limit exceeded. Automatically switching to developer endpoint (lldev.thespacedevs.com)...")
                    self.base_url = "https://lldev.thespacedevs.com/2.2.0/launch/previous/"
                    data = self.fetch_launches_page(
                        offset=offset,
                        limit=LL2_PAGE_LIMIT,
                        start_year=start_year,
                        end_year=end_year
                    )
                else:
                    raise err
            
            if total_count is None:
                total_count = data.get("count", 0)
                logger.info(f"Total available launches reported by API: {total_count}")

            results = data.get("results", [])
            if not results:
                logger.info("No more records returned. Pagination complete.")
                break

            for item in results:
                parsed = self._parse_launch_record(item)
                if parsed:
                    # Optional year filtering
                    year = pd.to_datetime(parsed["launch_time_utc"]).year
                    if start_year and year < start_year:
                        continue
                    if end_year and year > end_year:
                        continue
                    records.append(parsed)

            total_fetched += len(results)
            offset += len(results)

            # Auto-save checkpoint every 500 records to protect against network drops
            if save_intermediate and len(records) > 0 and (len(records) % 500 == 0 or offset >= (total_count or 0)):
                checkpoint_df = pd.DataFrame(records)
                cache_file = RAW_LL2_DIR / "ll2_launches_extracted.parquet"
                checkpoint_df.to_parquet(cache_file, index=False)
                logger.info(f"Checkpoint saved: {len(checkpoint_df)} records cached.")

            if max_records and len(records) >= max_records:
                logger.info(f"Reached requested limit of {max_records} records.")
                records = records[:max_records]
                break

            if total_count is not None and offset >= total_count:
                logger.info("Reached end of LL2 dataset.")
                break

            time.sleep(self.sleep_sec)

        df = pd.DataFrame(records)
        logger.info(f"Successfully processed {len(df)} valid launch records.")

        if save_intermediate and not df.empty:
            cache_file = RAW_LL2_DIR / "ll2_launches_extracted.parquet"
            df.to_parquet(cache_file, index=False)
            logger.info(f"Saved intermediate LL2 raw dataset to {cache_file}")

        return df

    def _parse_launch_record(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parses nested LL2 JSON into normalized dictionary aligning with target schema."""
        # Check target launch status
        status_info = item.get("status", {})
        status_id = status_info.get("id")

        if status_id in LL2_SUCCESS_STATUS_IDS:
            target = 1
        elif status_id in LL2_FAILURE_STATUS_IDS:
            target = 0
        else:
            # Skip scheduled / unknown / in-flight states
            return None

        # Coordinates
        pad = item.get("pad") or {}
        lat_str = pad.get("latitude")
        lon_str = pad.get("longitude")
        if not lat_str or not lon_str:
            return None

        try:
            latitude = float(lat_str)
            longitude = float(lon_str)
        except (ValueError, TypeError):
            return None

        launch_time = item.get("net")
        if not launch_time:
            return None

        # Rocket specifications
        rocket = item.get("rocket") or {}
        config = rocket.get("configuration") or {}
        rocket_id = config.get("id", "unknown")
        rocket_name = config.get("full_name") or config.get("name", "Unknown Rocket")
        
        # Categorize rocket type (Small, Medium, Heavy, Super Heavy)
        rocket_type = self._infer_rocket_type(config)

        # Mass & Thrust
        # In LL2, launch_mass is often metric tonnes or kg depending on payload/config
        total_mass_raw = config.get("launch_mass")
        total_mass_kg = self._normalize_mass(total_mass_raw)

        liftoff_thrust_kn = config.get("to_thrust")
        if liftoff_thrust_kn is not None:
            try:
                liftoff_thrust_kn = float(liftoff_thrust_kn)
            except (ValueError, TypeError):
                liftoff_thrust_kn = None

        # Structural Max Q Limit estimation (kPa)
        max_q_kpa = self._estimate_max_q(rocket_name)

        # Payload Mass (kg)
        mission = item.get("mission") or {}
        payloads = item.get("payloads") or []
        payload_mass_kg = 0.0
        for p in payloads:
            mass = p.get("mass")
            if mass:
                try:
                    payload_mass_kg += float(mass)
                except (ValueError, TypeError):
                    pass
        if payload_mass_kg == 0.0:
            payload_mass_kg = None

        return {
            "launch_id": item.get("id"),
            "launch_name": item.get("name"),
            "launch_time_utc": launch_time,
            "pad_name": pad.get("name", "Unknown Pad"),
            "latitude": latitude,
            "longitude": longitude,
            "rocket_id": rocket_id,
            "rocket_name": rocket_name,
            "rocket_type": rocket_type,
            "total_mass_kg": total_mass_kg,
            "liftoff_thrust_kn": liftoff_thrust_kn,
            "max_q_limit_kpa": max_q_kpa,
            "payload_mass_kg": payload_mass_kg,
            "launch_status": target
        }

    def _infer_rocket_type(self, config: Dict[str, Any]) -> str:
        """Categorizes rocket into Small, Medium, Heavy, Super Heavy based on payload to LEO/thrust."""
        leo_capacity = config.get("leo_capacity")
        if leo_capacity:
            try:
                leo = float(leo_capacity)
                if leo < 2000:
                    return "Small"
                elif leo <= 20000:
                    return "Medium"
                elif leo <= 50000:
                    return "Heavy"
                else:
                    return "Super Heavy"
            except (ValueError, TypeError):
                pass

        name = (config.get("name") or "").lower()
        if "heavy" in name:
            return "Heavy"
        if any(sm in name for sm in ["electron", "pegasus", "alpha", "hyperbola"]):
            return "Small"
        return "Medium"

    def _normalize_mass(self, raw_val: Any) -> Optional[float]:
        """Converts mass to kg safely."""
        if raw_val is None:
            return None
        try:
            val = float(raw_val)
            # If value is less than 5000, it's likely metric tonnes
            if 0 < val < 5000:
                return val * 1000.0
            return val
        except (ValueError, TypeError):
            return None

    def _estimate_max_q(self, rocket_name: str) -> float:
        name_lower = rocket_name.lower()
        for family, limit in DEFAULT_MAX_Q_LIMIT_KPA.items():
            if family in name_lower:
                return limit
        return DEFAULT_MAX_Q_LIMIT_KPA["default"]
