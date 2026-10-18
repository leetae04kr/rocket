import numpy as np
from typing import Tuple, Optional


def calculate_surface_wind_speed(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    """Calculates horizontal wind speed from 10m u and v vector components (m/s)."""
    return np.sqrt(np.square(u10) + np.square(v10))


def kelvin_to_celsius(temp_k: np.ndarray) -> np.ndarray:
    """Converts temperature from Kelvin to Celsius."""
    return temp_k - 273.15


def calculate_relative_humidity(t2m_k: np.ndarray, d2m_k: np.ndarray) -> np.ndarray:
    """
    Computes relative humidity (%) using the August-Roche-Magnus approximation
    derived from 2m Temperature (K) and 2m Dewpoint Temperature (K).
    """
    t_c = kelvin_to_celsius(t2m_k)
    td_c = kelvin_to_celsius(d2m_k)

    # Magnus constants
    a = 17.625
    b = 243.04

    # Saturation vapor pressures (hPa)
    es = 6.1078 * np.exp((a * t_c) / (b + t_c))
    e = 6.1078 * np.exp((a * td_c) / (b + td_c))

    rh = 100.0 * (e / es)
    return np.clip(rh, 0.0, 100.0)


def convert_precipitation_to_mm_hr(tp_m: np.ndarray) -> np.ndarray:
    """
    ERA5 hourly total precipitation (tp) is in meters accumulated over the hour.
    Converts meters/hour to mm/hour.
    """
    return np.maximum(tp_m * 1000.0, 0.0)


def estimate_visibility_km(rh_percent: np.ndarray, tp_mm_hr: np.ndarray) -> np.ndarray:
    """
    Empirical atmospheric visibility estimation based on relative humidity
    and precipitation rate (Koschmieder-type approximation for aviation).
    Standard clean air has ~40-50 km visibility, dropping exponentially under high humidity and rain.
    """
    # Extinction coefficient proxy (km^-1)
    # Background Rayleigh + aerosol extinction
    beta_bg = 0.02
    
    # Humidity hygroscopic aerosol swelling effect
    rh_norm = np.clip(rh_percent / 100.0, 0.0, 0.999)
    beta_humidity = 0.03 * (1.0 / (1.0 - rh_norm))**0.5
    
    # Rain extinction coefficient
    beta_rain = 0.25 * np.power(tp_mm_hr, 0.7)

    beta_total = beta_bg + beta_humidity + beta_rain
    
    # Koschmieder formula: V = 3.912 / beta
    vis = 3.912 / beta_total
    return np.clip(vis, 0.1, 50.0)


def compute_vertical_wind_shear(
    geopotential: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray
) -> float:
    """
    Computes maximum vertical wind shear (s^-1 = m/s per m) across vertical pressure levels.
    z = geopotential / 9.80665 (geopotential height in meters)
    Shear = |dV| / dz
    """
    z_m = geopotential / 9.80665
    
    # Sort by altitude ascending
    sort_idx = np.argsort(z_m)
    z_sorted = z_m[sort_idx]
    u_sorted = u_wind[sort_idx]
    v_sorted = v_wind[sort_idx]

    dz = np.diff(z_sorted)
    valid = dz > 10.0  # Avoid zero division or microscopic level steps

    du = np.diff(u_sorted)
    dv = np.diff(v_sorted)
    vector_dv = np.sqrt(du**2 + dv**2)

    shear_profile = np.zeros_like(dz)
    shear_profile[valid] = vector_dv[valid] / dz[valid]

    if len(shear_profile) == 0:
        return 0.0
    return float(np.max(shear_profile))


def interpolate_wind_at_altitudes(
    geopotential: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    target_altitudes_km: list = [5.0, 10.0, 20.0]
) -> dict:
    """
    Interpolates upper-air total horizontal wind speed (m/s) at specified target altitudes (km).
    """
    z_km = (geopotential / 9.80665) / 1000.0
    wind_speed = np.sqrt(np.square(u_wind) + np.square(v_wind))

    sort_idx = np.argsort(z_km)
    z_sorted = z_km[sort_idx]
    ws_sorted = wind_speed[sort_idx]

    results = {}
    for alt_km in target_altitudes_km:
        # np.interp linear interpolation
        interpolated_ws = np.interp(alt_km, z_sorted, ws_sorted, left=ws_sorted[0], right=ws_sorted[-1])
        results[f"upper_air_wind_speed_{int(alt_km)}km"] = float(interpolated_ws)

    return results


def calculate_freezing_level_altitude(
    geopotential: np.ndarray,
    temp_k: np.ndarray
) -> float:
    """
    Determines the freezing level altitude (km) where atmospheric temperature first drops below 273.15 K (0°C).
    Returns 0.0 if ground temperature is already below freezing.
    """
    z_km = (geopotential / 9.80665) / 1000.0
    t_c = temp_k - 273.15

    sort_idx = np.argsort(z_km)
    z_sorted = z_km[sort_idx]
    t_sorted = t_c[sort_idx]

    # If surface temperature is already freezing, altitude is 0 km
    if t_sorted[0] <= 0.0:
        return 0.0

    # Find first crossing below 0°C
    below_zero_indices = np.where(t_sorted <= 0.0)[0]
    if len(below_zero_indices) == 0:
        # Troposphere did not reach freezing within levels probed
        return float(z_sorted[-1])

    first_idx = below_zero_indices[0]
    if first_idx == 0:
        return float(z_sorted[0])

    # Linear interpolation between first_idx - 1 and first_idx
    z0, z1 = z_sorted[first_idx - 1], z_sorted[first_idx]
    t0, t1 = t_sorted[first_idx - 1], t_sorted[first_idx]

    if t1 == t0:
        return float(z0)

    # t(z) = t0 + (t1 - t0) * (z - z0)/(z1 - z0) = 0
    # z = z0 - t0 * (z1 - z0)/(t1 - t0)
    z_zero = z0 - t0 * (z1 - z0) / (t1 - t0)
    return float(np.clip(z_zero, 0.0, 25.0))
