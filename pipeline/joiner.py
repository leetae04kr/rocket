import gc
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import xarray as xr

from pipeline.physics import (
    calculate_surface_wind_speed,
    kelvin_to_celsius,
    calculate_relative_humidity,
    convert_precipitation_to_mm_hr,
    estimate_visibility_km,
    compute_vertical_wind_shear,
    interpolate_wind_at_altitudes,
    calculate_freezing_level_altitude
)
from pipeline.config import UPPER_AIR_PROBE_ALTITUDES_KM

logger = logging.getLogger("SpatioTemporalJoiner")


class SpatioTemporalJoiner:
    """
    Executes spatio-temporal nearest-neighbor joins between launch events and ERA5 grids using Xarray.
    Integrates physics-based feature engineering directly during extraction.
    """

    def __init__(self):
        pass

    def join_launches_with_era5(
        self,
        launches_df: pd.DataFrame,
        surface_nc_path: Path,
        pressure_nc_path: Path
    ) -> pd.DataFrame:
        """
        Takes a subset/batch of launch records and matches each with the closest grid cell and timestamp
        in the provided surface and pressure NetCDF files.
        """
        if launches_df.empty:
            return pd.DataFrame()

        logger.info(f"Joining {len(launches_df)} launches with ERA5 datasets...")
        enriched_records: List[Dict[str, Any]] = []

        with xr.open_dataset(surface_nc_path) as ds_surf, xr.open_dataset(pressure_nc_path) as ds_press:
            # Check coordinate conventions (e.g. 0..360 vs -180..180)
            surf_lon_min = float(ds_surf.coords["longitude"].min())
            press_lon_min = float(ds_press.coords["longitude"].min())

            for _, row in launches_df.iterrows():
                try:
                    record_dict = row.to_dict()
                    launch_dt = pd.to_datetime(row["launch_time_utc"])
                    # Normalize timezone: ERA5 NetCDF time coordinate is typically tz-naive UTC
                    if launch_dt.tzinfo is not None:
                        launch_dt = launch_dt.tz_convert("UTC").tz_localize(None)

                    lat = float(row["latitude"])
                    lon = float(row["longitude"])

                    # Coordinate adaptation for longitude if ERA5 is 0..360
                    surf_lon = lon if surf_lon_min < 0 else (lon % 360)
                    press_lon = lon if press_lon_min < 0 else (lon % 360)

                    # Surface nearest neighbor extraction
                    pt_surf = ds_surf.sel(
                        time=launch_dt,
                        latitude=lat,
                        longitude=surf_lon,
                        method="nearest"
                    )

                    u10 = float(pt_surf["u10"].values)
                    v10 = float(pt_surf["v10"].values)
                    fg10 = float(pt_surf["fg10"].values) if "fg10" in pt_surf else np.sqrt(u10**2 + v10**2) * 1.3
                    t2m = float(pt_surf["t2m"].values)
                    d2m = float(pt_surf["d2m"].values)
                    tp = float(pt_surf["tp"].values) if "tp" in pt_surf else 0.0

                    # Surface physical derivations
                    sfc_wind_spd = calculate_surface_wind_speed(u10, v10)
                    sfc_temp_c = kelvin_to_celsius(t2m)
                    sfc_rh = calculate_relative_humidity(t2m, d2m)
                    precip_mm_hr = convert_precipitation_to_mm_hr(tp)
                    visibility = estimate_visibility_km(sfc_rh, precip_mm_hr)

                    # Upper-air profile nearest neighbor extraction
                    pt_press = ds_press.sel(
                        time=launch_dt,
                        latitude=lat,
                        longitude=press_lon,
                        method="nearest"
                    )

                    z_profile = np.asarray(pt_press["z"].values, dtype=np.float64)
                    u_profile = np.asarray(pt_press["u"].values, dtype=np.float64)
                    v_profile = np.asarray(pt_press["v"].values, dtype=np.float64)
                    t_profile = np.asarray(pt_press["t"].values, dtype=np.float64)

                    # Upper-air derivations
                    max_shear = compute_vertical_wind_shear(z_profile, u_profile, v_profile)
                    upper_winds = interpolate_wind_at_altitudes(
                        z_profile, u_profile, v_profile, UPPER_AIR_PROBE_ALTITUDES_KM
                    )
                    freezing_alt = calculate_freezing_level_altitude(z_profile, t_profile)

                    # Merge features
                    record_dict.update({
                        "surface_wind_speed": round(float(sfc_wind_spd), 2),
                        "wind_gust": round(float(fg10), 2),
                        "surface_temperature": round(float(sfc_temp_c), 2),
                        "surface_humidity": round(float(sfc_rh), 2),
                        "precipitation": round(float(precip_mm_hr), 4),
                        "visibility": round(float(visibility), 2),
                        "max_wind_shear": round(float(max_shear), 5),
                        "freezing_level_altitude_km": round(float(freezing_alt), 2),
                        **{k: round(v, 2) for k, v in upper_winds.items()}
                    })

                    enriched_records.append(record_dict)

                except Exception as e:
                    logger.warning(f"Error processing launch {row.get('launch_id')}: {e}")
                    continue

        gc.collect()
        joined_df = pd.DataFrame(enriched_records)
        logger.info(f"Successfully joined {len(joined_df)} records.")
        return joined_df
