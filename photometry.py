"""Convert WD mass/temperature and MS mass/age/metallicity to Gaia magnitudes."""

import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator


def component_photometry(rows, wd_models_path, atmosphere="H"):
    """Return CMD labels; masses are current masses in solar units, ages in Gyr."""
    sys.path.insert(0, str(Path(wd_models_path).expanduser().resolve().parent))
    wd = importlib.import_module("WD_models")
    mass, logg, age, cooling_age, logteff, mbol = wd.read_cooling_tracks(
        "Bedard2020", "Bedard2020", "ONe", atmosphere)
    mass_teff = np.column_stack([mass, logteff])
    inputs = np.array([[float(r["mass_wd"]), np.log10(float(r["teff_wd"]))] for r in rows])
    logg_input = LinearNDInterpolator(mass_teff, logg, rescale=True)(inputs)
    mbol_input = LinearNDInterpolator(mass_teff, mbol, rescale=True)(inputs)
    wd_mags = []
    for band in ["bp3", "G3", "rp3"]:
        _, correction = wd.interp_atm(atmosphere, band + "-Mbol")
        wd_mags.append(correction(inputs[:, 1], logg_input) + mbol_input)
    wd_mags = np.column_stack(wd_mags)
    with np.load(Path(__file__).parent / "models" / "parsec_ms.npz") as p:
        grid = np.column_stack([p["mass"], p["mh"], p["logage"]])
        ms = LinearNDInterpolator(grid, p["mags"], rescale=True)
    ms_inputs = np.array([[float(r["mass_ms"]), float(r["mh_ms"]),
                           np.log10(float(r["age_ms_gyr"]) * 1e9)] for r in rows])
    ms_mags = ms(ms_inputs)

    def labels(mags):
        return np.column_stack([mags[:, 0] - mags[:, 2], mags[:, 1], mags[:, 2]])

    return labels(wd_mags), labels(ms_mags)
