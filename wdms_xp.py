"""Gaia XP synthesis and WD+MS fitting in component CMD coordinates."""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


MODEL_FILE = Path(__file__).parent / "models" / "xp_model.npz"
LABELS = ("bp_rp", "abs_g", "abs_rp")


class XPModel:
    def __init__(self, filename=MODEL_FILE):
        with np.load(filename) as data:
            self.data = {k: data[k] for k in data.files}
        self.wave_nm = self.data["wave_nm"]
        self.label_center = self.data["label_mean"]
        self.label_scale = self.data["label_scale"]
        self.flux_center = self.data["flux_mean"]
        self.flux_scale = self.data["flux_scale"]
        self.binary_bounds = np.array([
            [-3.710272, -3.607852, -1.9825581, -1.3816054, -5.189663, -1.5048838],
            [1.7231159, 3.5154874, 3.6297603, 1.8675938, 2.3340635, 3.0325983],
        ])
        self.single_bounds = np.array([
            [-3.710272, -5.189663, -1.9825581],
            [1.8675938, 3.5154874, 3.6297603],
        ])

    def scale_labels(self, labels):
        return (np.asarray(labels) - self.label_center) / self.label_scale

    def unscale_labels(self, labels):
        return np.asarray(labels) * self.label_scale + self.label_center

    def scaled_spectrum(self, scaled_labels):
        d = self.data
        x = (np.asarray(scaled_labels, dtype=float) - d["xmin"]) / (
            d["xmax"] - d["xmin"]) - 0.5
        for i in range(5):
            x = d[f"w_{i}"] @ x + d[f"b_{i}"]
            x = np.where(x >= 0, x, 0.01 * x)
        return (d["w_5"] @ x + d["b_5"]) * d["yscale"]

    def spectrum(self, labels):
        """Component (BP-RP, M_G, M_RP) -> linear flux at 10 pc."""
        return self.scaled_spectrum(self.scale_labels(labels)) * self.flux_scale + self.flux_center

    def binary_spectrum(self, wd_labels, ms_labels):
        return self.spectrum(wd_labels) + self.spectrum(ms_labels)

    def in_bounds(self, wd_labels, ms_labels):
        x = np.r_[self.scale_labels(wd_labels), self.scale_labels(ms_labels)]
        return bool(np.all(np.isfinite(x)) and np.all(x >= self.binary_bounds[0])
                    and np.all(x <= self.binary_bounds[1]))

    def _initial(self, scaled_flux, prefix):
        x = np.asarray(scaled_flux, dtype=np.float32)
        for i in range(6):
            x = self.data[f"{prefix}_w_{i}"] @ x + self.data[f"{prefix}_b_{i}"]
            if i < 5:
                x = np.where(x >= 0, x, 0.01 * x)
        return x

    def fit(self, flux, sigma):
        """Fit both hypotheses with the same flux scale and error vector."""
        flux, sigma = np.asarray(flux), np.asarray(sigma)
        valid = np.isfinite(flux) & np.isfinite(sigma) & (sigma > 0)
        if valid.sum() <= 6:
            raise ValueError("More than six valid XP pixels are required.")
        y = (flux - self.flux_center) / self.flux_scale
        err = sigma / self.flux_scale
        initial_flux = np.where(valid, y, 0.0)
        results = {}
        for hypothesis, bounds in [("single", self.single_bounds), ("binary", self.binary_bounds)]:
            p0 = np.clip(self._initial(initial_flux, hypothesis + "_init"), *bounds)

            def predict(x):
                if hypothesis == "single":
                    return self.scaled_spectrum(x)
                f = self.binary_spectrum(self.unscale_labels(x[:3]), self.unscale_labels(x[3:]))
                return (f - self.flux_center) / self.flux_scale

            def residual(x):
                return (predict(x)[valid] - y[valid]) / err[valid]

            fit = least_squares(residual, p0, bounds=bounds)
            chi2 = float(np.sum(fit.fun**2))
            results[f"chi2_{hypothesis}"] = chi2
            results[f"chi2_{hypothesis}_reduced"] = chi2 / (valid.sum() - len(fit.x))
            results[f"success_{hypothesis}"] = bool(fit.success)
            results[f"nfev_{hypothesis}"] = int(fit.nfev)
            results[f"flux_{hypothesis}"] = predict(fit.x) * self.flux_scale + self.flux_center
            labels = self.unscale_labels(fit.x) if hypothesis == "single" else np.r_[
                self.unscale_labels(fit.x[:3]), self.unscale_labels(fit.x[3:])]
            results[f"labels_{hypothesis}"] = labels
        results["n_valid"] = int(valid.sum())
        return results


def add_noise(flux, snr, rng):
    """Add independent Gaussian noise; return its standard deviation."""
    if not np.isfinite(snr) or snr <= 0:
        raise ValueError("SNR must be positive.")
    sigma = np.abs(np.asarray(flux)) / snr
    return flux + rng.normal(size=sigma.shape) * sigma, sigma


def selection(fit, chi2_max=20.0, chi2_kind="reduced"):
    """Evaluate the published table's chi-square, contrast, and WD CMD cuts."""
    wd, ms = fit["labels_binary"][:3], fit["labels_binary"][3:]
    suffix = "_reduced" if chi2_kind == "reduced" else ""
    flags = {
        "pass_quality": bool(fit[f"chi2_single{suffix}"] < chi2_max
                             and fit[f"chi2_binary{suffix}"] < chi2_max),
        "pass_improvement": fit["chi2_binary"] < fit["chi2_single"],
        "pass_contrast_g": bool(abs(wd[1] - ms[1]) <= 2.5),
        "pass_contrast_rp": bool(abs(wd[2] - ms[2]) <= 2.5),
        "pass_wd_cmd": bool(wd[1] > 9 + 2.5 * wd[0]),
        "fit_converged": fit["success_single"] and fit["success_binary"],
    }
    flags["pass_selection"] = all(flags[k] for k in flags if k.startswith("pass_"))
    return flags
