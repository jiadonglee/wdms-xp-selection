"""Check binary flux recovery and the statistical meaning of mock errors."""

import numpy as np

from wdms_xp import XPModel, add_noise


model = XPModel()
wd = [-0.038, 11.147, 11.108]
ms = [2.99, 10.924, 9.58]
truth = model.binary_spectrum(wd, ms)
fit = model.fit(truth, np.abs(truth) / 20)
assert fit["success_binary"]
assert fit["chi2_binary"] < 1e-6, fit["chi2_binary"]
assert fit["chi2_single"] > fit["chi2_binary"] + 1

flux = np.full(10000, 100.0)
observed, sigma = add_noise(flux, 20, np.random.default_rng(42))
assert np.all(sigma > 0)
residual = (observed - flux) / sigma
assert abs(residual.mean()) < 0.03
assert abs(residual.std() - 1) < 0.03
print("Binary recovery and Gaussian error scaling passed.")
