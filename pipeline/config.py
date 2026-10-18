import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_LL2_DIR = DATA_DIR / "raw_ll2"
RAW_ERA5_DIR = DATA_DIR / "raw_era5"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

for d in [DATA_DIR, RAW_LL2_DIR, RAW_ERA5_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# LL2 (Launch Library 2) Configuration
# Default to developer endpoint (lldev.thespacedevs.com) which has much higher rate limits and no IP lockouts
LL2_BASE_URL = os.getenv("LL2_BASE_URL", "https://lldev.thespacedevs.com/2.2.0/launch/previous/")
LL2_RATE_LIMIT_SLEEP_SEC = float(os.getenv("LL2_SLEEP_SEC", "3.0"))  # SpaceDevs allows ~15 req/min on free tier
LL2_MAX_RETRIES = 5
LL2_TIMEOUT_SEC = 30
LL2_PAGE_LIMIT = 100  # API max is typically 100

# Launch Status Target Mapping
# LL2 status IDs:
# 1: Go for Launch
# 2: To Be Determined
# 3: Launch Successful -> Target 1
# 4: Launch Failure -> Target 0
# 5: Unplanned Hold
# 6: In Flight
# 7: Partial Failure -> Target 0
LL2_SUCCESS_STATUS_IDS = {3}
LL2_FAILURE_STATUS_IDS = {4, 7}

# Standard Structural Max Q limits by Rocket Class / Family (in kPa)
# Standard orbital launch vehicles typically experience peak dynamic pressure between 30 and 42 kPa.
DEFAULT_MAX_Q_LIMIT_KPA = {
    "falcon 9": 35.0,
    "falcon heavy": 38.0,
    "atlas v": 39.0,
    "delta iv": 37.0,
    "electron": 32.0,
    "ariane 5": 40.0,
    "space shuttle": 33.0,
    "soyuz": 36.0,
    "long march": 35.0,
    "h-iia": 38.0,
    "default": 35.0
}

# ERA5 Pressure Levels (hPa) covering surface (~1000) up to stratosphere (~50 hPa, ~20 km)
ERA5_PRESSURE_LEVELS = [
    "50", "100", "200", "300", "400", "500", "600", "700", "850", "925", "1000"
]

# High-altitude wind speed probe heights in kilometers
UPPER_AIR_PROBE_ALTITUDES_KM = [5.0, 10.0, 20.0]

# Output final dataset columns schema
DATASET_SCHEMA = [
    # Metadata
    "launch_id",
    "launch_name",
    "launch_time_utc",
    "pad_name",
    "latitude",
    "longitude",
    # Rocket Specifications
    "rocket_id",
    "rocket_name",
    "rocket_type",
    "total_mass_kg",
    "liftoff_thrust_kn",
    "max_q_limit_kpa",
    "payload_mass_kg",
    # Surface Weather (ERA5)
    "surface_wind_speed",         # m/s
    "wind_gust",                  # m/s
    "surface_temperature",        # °C
    "surface_humidity",           # %
    "precipitation",              # mm/hr
    "visibility",                 # km (approximated / relative humidity proxy)
    # Upper-Air Weather (ERA5)
    "max_wind_shear",             # s^-1 (m/s per m)
    "upper_air_wind_speed_5km",   # m/s
    "upper_air_wind_speed_10km",  # m/s
    "upper_air_wind_speed_20km",  # m/s
    "freezing_level_altitude_km", # km
    # Target
    "launch_status"               # 1 (Success), 0 (Failure)
]
