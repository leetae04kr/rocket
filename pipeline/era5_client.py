import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from pipeline.config import (
    RAW_ERA5_DIR,
    ERA5_PRESSURE_LEVELS
)

logger = logging.getLogger("ERA5Client")


class ERA5BatchDownloader:
    """
    ECMWF Climate Data Store (CDS) API batch orchestrator for ERA5 reanalysis data.
    Implements monthly/yearly chunk requests and NetCDF caching to avoid excessive CDS queue times.
    """

    def __init__(self, cache_dir: Path = RAW_ERA5_DIR):
        self.cache_dir = cache_dir
        self.client = self._init_cds_client()

    def _init_cds_client(self):
        """Attempts to initialize cdsapi.Client; gracefully handles missing credentials."""
        try:
            import cdsapi
            client = cdsapi.Client()
            logger.info("Successfully authenticated with ECMWF CDS API.")
            return client
        except Exception as e:
            logger.warning(
                f"ECMWF cdsapi client could not be initialized ({e}). "
                "Ensure ~/.cdsapirc file or CDSAPI_URL and CDSAPI_KEY env vars are configured. "
                "The pipeline can operate in synthetic/offline mode if required."
            )
            return None

    def download_surface_batch(
        self,
        year: int,
        month: int,
        days: List[str],
        times: List[str],
        area_bbox: Optional[List[float]] = None,
        force_redownload: bool = False
    ) -> Path:
        """
        Requests ERA5 Single Levels (Surface) reanalysis data for a specific year and month.
        Variables: 10m u/v wind, wind gust, 2m temp, 2m dewpoint, total precipitation.
        BBox format: [North, West, South, East]
        """
        file_name = f"era5_surface_{year}_{month:02d}.nc"
        target_path = self.cache_dir / file_name

        if target_path.exists() and not force_redownload:
            logger.info(f"Using cached ERA5 surface data: {target_path}")
            return target_path

        if self.client is None:
            logger.warning(f"CDS API unavailable. Generating synthetic surface NetCDF for {year}-{month:02d} for local test/verification.")
            return self._generate_synthetic_surface_nc(target_path, year, month, days, times)

        request_payload = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "instantaneous_10m_wind_gust",
                "2m_temperature",
                "2m_dewpoint_temperature",
                "total_precipitation"
            ],
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": times
        }

        if area_bbox:
            request_payload["area"] = area_bbox  # [N, W, S, E]

        logger.info(f"Submitting CDS request for surface data: {year}-{month:02d} (days: {len(days)})")
        self.client.retrieve("reanalysis-era5-single-levels", request_payload, str(target_path))
        logger.info(f"Downloaded ERA5 surface data saved to {target_path}")
        return target_path

    def download_pressure_batch(
        self,
        year: int,
        month: int,
        days: List[str],
        times: List[str],
        area_bbox: Optional[List[float]] = None,
        force_redownload: bool = False
    ) -> Path:
        """
        Requests ERA5 Pressure Levels (Upper Air) reanalysis data for a specific year and month.
        Variables: u, v, geopotential (z), temperature (t) across standard atmospheric levels.
        """
        file_name = f"era5_pressure_{year}_{month:02d}.nc"
        target_path = self.cache_dir / file_name

        if target_path.exists() and not force_redownload:
            logger.info(f"Using cached ERA5 pressure data: {target_path}")
            return target_path

        if self.client is None:
            logger.warning(f"CDS API unavailable. Generating synthetic pressure NetCDF for {year}-{month:02d} for local test/verification.")
            return self._generate_synthetic_pressure_nc(target_path, year, month, days, times)

        request_payload = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind",
                "geopotential",
                "temperature"
            ],
            "pressure_level": ERA5_PRESSURE_LEVELS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": times
        }

        if area_bbox:
            request_payload["area"] = area_bbox

        logger.info(f"Submitting CDS request for pressure levels: {year}-{month:02d}")
        self.client.retrieve("reanalysis-era5-pressure-levels", request_payload, str(target_path))
        logger.info(f"Downloaded ERA5 pressure data saved to {target_path}")
        return target_path

    def _generate_synthetic_surface_nc(self, target_path: Path, year: int, month: int, days: List[str], times: List[str]) -> Path:
        """Creates physically realistic mock NetCDF dataset when CDS credentials are not present."""
        import xarray as xr
        
        # Build datetime index
        date_strs = [f"{year}-{month:02d}-{int(d):02d}T{t}" for d in days for t in times]
        time_index = pd.to_datetime(date_strs).unique()

        lats = np.linspace(-60, 60, 5)
        lons = np.linspace(-180, 180, 5)

        shape = (len(time_index), len(lats), len(lons))
        np.random.seed(42 + month)

        ds = xr.Dataset(
            {
                "u10": (["time", "latitude", "longitude"], np.random.normal(3.0, 5.0, shape).astype(np.float32)),
                "v10": (["time", "latitude", "longitude"], np.random.normal(2.0, 4.0, shape).astype(np.float32)),
                "fg10": (["time", "latitude", "longitude"], np.abs(np.random.normal(8.0, 4.0, shape)).astype(np.float32)),
                "t2m": (["time", "latitude", "longitude"], (np.random.normal(293.15, 8.0, shape)).astype(np.float32)),
                "d2m": (["time", "latitude", "longitude"], (np.random.normal(285.15, 6.0, shape)).astype(np.float32)),
                "tp": (["time", "latitude", "longitude"], (np.maximum(0, np.random.normal(0.0005, 0.002, shape))).astype(np.float32)),
            },
            coords={"time": time_index, "latitude": lats, "longitude": lons}
        )
        ds.to_netcdf(target_path)
        logger.info(f"Generated synthetic ERA5 surface file at {target_path}")
        return target_path

    def _generate_synthetic_pressure_nc(self, target_path: Path, year: int, month: int, days: List[str], times: List[str]) -> Path:
        """Creates physically realistic mock Upper-Air Pressure Levels NetCDF dataset."""
        import xarray as xr

        date_strs = [f"{year}-{month:02d}-{int(d):02d}T{t}" for d in days for t in times]
        time_index = pd.to_datetime(date_strs).unique()

        levels = [int(p) for p in ERA5_PRESSURE_LEVELS]
        levels.sort()
        lats = np.linspace(-60, 60, 5)
        lons = np.linspace(-180, 180, 5)

        shape = (len(time_index), len(levels), len(lats), len(lons))
        np.random.seed(100 + month)

        # Standard geopotential approx (m^2/s^2)
        # 1000 hPa ~ 100m, 50 hPa ~ 20,500m
        z_base = np.array([
            20500 * 9.80665, 16000 * 9.80665, 12000 * 9.80665, 9000 * 9.80665,
            7000 * 9.80665, 5500 * 9.80665, 4200 * 9.80665, 3000 * 9.80665,
            1500 * 9.80665, 800 * 9.80665, 100 * 9.80665
        ], dtype=np.float32)

        z_4d = np.tile(z_base[None, :, None, None], (len(time_index), 1, len(lats), len(lons)))
        # Temperature profile (Troposphere lapse rate approx)
        t_base = np.array([215, 210, 218, 235, 250, 260, 270, 278, 285, 289, 292], dtype=np.float32)
        t_4d = np.tile(t_base[None, :, None, None], (len(time_index), 1, len(lats), len(lons)))

        ds = xr.Dataset(
            {
                "u": (["time", "level", "latitude", "longitude"], np.random.normal(15.0, 10.0, shape).astype(np.float32)),
                "v": (["time", "level", "latitude", "longitude"], np.random.normal(5.0, 8.0, shape).astype(np.float32)),
                "z": (["time", "level", "latitude", "longitude"], z_4d),
                "t": (["time", "level", "latitude", "longitude"], t_4d),
            },
            coords={"time": time_index, "level": levels, "latitude": lats, "longitude": lons}
        )
        ds.to_netcdf(target_path)
        logger.info(f"Generated synthetic ERA5 pressure levels file at {target_path}")
        return target_path
