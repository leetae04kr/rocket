import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import xarray as xr

from pipeline.physics import (
    calculate_surface_wind_speed,
    calculate_relative_humidity,
    compute_vertical_wind_shear,
    interpolate_wind_at_altitudes,
    calculate_freezing_level_altitude
)
from pipeline.ll2_client import LL2DataCollector
from pipeline.joiner import SpatioTemporalJoiner
from pipeline.config import DATASET_SCHEMA


class TestRocketAtmospherePipeline(unittest.TestCase):

    def test_physics_surface_wind(self):
        u = np.array([3.0, 0.0])
        v = np.array([4.0, 0.0])
        spd = calculate_surface_wind_speed(u, v)
        self.assertAlmostEqual(spd[0], 5.0)
        self.assertAlmostEqual(spd[1], 0.0)

    def test_physics_relative_humidity(self):
        # When T = Td, RH should be 100%
        t_k = np.array([293.15])
        d_k = np.array([293.15])
        rh = calculate_relative_humidity(t_k, d_k)
        self.assertAlmostEqual(rh[0], 100.0, places=1)

    def test_physics_wind_shear_and_freezing_level(self):
        # 3 atmospheric levels
        # z: 0m, 5000m, 10000m
        z = np.array([0.0, 5000.0 * 9.80665, 10000.0 * 9.80665])
        u = np.array([5.0, 15.0, 35.0])
        v = np.array([0.0, 0.0, 0.0])
        t = np.array([288.15, 260.0, 230.0])  # +15C, -13.15C, -43.15C

        shear = compute_vertical_wind_shear(z, u, v)
        self.assertGreater(shear, 0.0)

        winds = interpolate_wind_at_altitudes(z, u, v, [5.0, 10.0])
        self.assertAlmostEqual(winds["upper_air_wind_speed_5km"], 15.0, places=1)
        self.assertAlmostEqual(winds["upper_air_wind_speed_10km"], 35.0, places=1)

        freezing_alt = calculate_freezing_level_altitude(z, t)
        # Should be between 0 and 5 km
        self.assertGreater(freezing_alt, 0.0)
        self.assertLess(freezing_alt, 5.0)

    def test_spatio_temporal_joiner_with_mock_netcdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            surf_file = tmp_path / "mock_surf.nc"
            press_file = tmp_path / "mock_press.nc"

            times = pd.date_range("2023-01-01 00:00", periods=4, freq="6h")
            lats = [28.5]  # Cape Canaveral
            lons = [-80.5]
            levels = [1000, 500, 200]

            # Surface Dataset
            shape_s = (len(times), len(lats), len(lons))
            ds_s = xr.Dataset(
                {
                    "u10": (["time", "latitude", "longitude"], np.full(shape_s, 4.0)),
                    "v10": (["time", "latitude", "longitude"], np.full(shape_s, 3.0)),
                    "fg10": (["time", "latitude", "longitude"], np.full(shape_s, 8.0)),
                    "t2m": (["time", "latitude", "longitude"], np.full(shape_s, 295.15)),
                    "d2m": (["time", "latitude", "longitude"], np.full(shape_s, 288.15)),
                    "tp": (["time", "latitude", "longitude"], np.full(shape_s, 0.001)),
                },
                coords={"time": times, "latitude": lats, "longitude": lons}
            )
            ds_s.to_netcdf(surf_file)

            # Pressure Dataset
            shape_p = (len(times), len(levels), len(lats), len(lons))
            z_vals = np.array([100 * 9.80665, 5500 * 9.80665, 12000 * 9.80665])
            z_4d = np.tile(z_vals[None, :, None, None], (len(times), 1, len(lats), len(lons)))
            t_vals = np.array([293.0, 255.0, 220.0])
            t_4d = np.tile(t_vals[None, :, None, None], (len(times), 1, len(lats), len(lons)))

            ds_p = xr.Dataset(
                {
                    "u": (["time", "level", "latitude", "longitude"], np.full(shape_p, 10.0)),
                    "v": (["time", "level", "latitude", "longitude"], np.full(shape_p, 5.0)),
                    "z": (["time", "level", "latitude", "longitude"], z_4d),
                    "t": (["time", "level", "latitude", "longitude"], t_4d),
                },
                coords={"time": times, "level": levels, "latitude": lats, "longitude": lons}
            )
            ds_p.to_netcdf(press_file)

            # Mock launch records
            launches_df = pd.DataFrame([
                {
                    "launch_id": "test-launch-001",
                    "launch_name": "Falcon 9 Test",
                    "launch_time_utc": "2023-01-01T06:00:00Z",
                    "pad_name": "LC-39A",
                    "latitude": 28.57,
                    "longitude": -80.58,
                    "rocket_id": "falcon9",
                    "rocket_name": "Falcon 9 Block 5",
                    "rocket_type": "Medium",
                    "total_mass_kg": 549054.0,
                    "liftoff_thrust_kn": 7607.0,
                    "max_q_limit_kpa": 35.0,
                    "payload_mass_kg": 15000.0,
                    "launch_status": 1
                }
            ])

            joiner = SpatioTemporalJoiner()
            joined = joiner.join_launches_with_era5(launches_df, surf_file, press_file)

            self.assertEqual(len(joined), 1)
            self.assertIn("surface_wind_speed", joined.columns)
            self.assertAlmostEqual(joined.iloc[0]["surface_wind_speed"], 5.0)
            self.assertIn("max_wind_shear", joined.columns)
            self.assertIn("freezing_level_altitude_km", joined.columns)
            self.assertEqual(joined.iloc[0]["launch_status"], 1)


if __name__ == "__main__":
    unittest.main()
