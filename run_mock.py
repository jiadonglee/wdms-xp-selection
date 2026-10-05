"""Run a simulated WD+MS CSV through XP synthesis, fitting, and selection."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photometry import component_photometry
from wdms_xp import XPModel, add_noise, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--snr", type=float, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wd-models", type=Path, default=Path("external/WD_models"))
    parser.add_argument("--atmosphere", choices=["H", "He"], default="H")
    parser.add_argument("--chi2-kind", choices=["reduced", "total"], default="reduced")
    parser.add_argument("--chi2-max", type=float, default=20)
    args = parser.parse_args()
    with args.input.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input CSV is empty.")
    if "bp_rp_wd" in rows[0]:
        wd_labels = np.array([[float(r[k]) for k in ("bp_rp_wd", "abs_g_wd", "abs_rp_wd")] for r in rows])
        ms_labels = np.array([[float(r[k]) for k in ("bp_rp_ms", "abs_g_ms", "abs_rp_ms")] for r in rows])
    else:
        wd_labels, ms_labels = component_photometry(rows, args.wd_models, args.atmosphere)
    model, rng = XPModel(), np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    results, spectra, indices, fits = [], [], [], []
    for i, (row, wd, ms) in enumerate(zip(rows, wd_labels, ms_labels)):
        output = dict(row)
        for component, labels in [("wd", wd), ("ms", ms)]:
            for name, value in zip(("bp_rp", "abs_g", "abs_rp"), labels):
                output[f"{name}_{component}"] = value
        output["status"] = "ok"
        if not np.all(np.isfinite(np.r_[wd, ms])):
            output["status"] = "outside_photometry_grid"
        elif not model.in_bounds(wd, ms):
            output["status"] = "outside_xp_bounds"
        if output["status"] != "ok":
            results.append(output)
            continue
        truth = model.binary_spectrum(wd, ms)
        observed, sigma = add_noise(truth, args.snr, rng)
        fit = model.fit(observed, sigma)
        output.update({k: v for k, v in fit.items() if not k.startswith(("flux_", "labels_"))})
        output.update(selection(fit, args.chi2_max, args.chi2_kind))
        output.update(snr=args.snr, chi2_kind=args.chi2_kind, chi2_max=args.chi2_max)
        for component, labels in [("wd", fit["labels_binary"][:3]), ("ms", fit["labels_binary"][3:])]:
            for name, value in zip(("bp_rp", "abs_g", "abs_rp"), labels):
                output[f"fit_{name}_{component}"] = value
        results.append(output)
        indices.append(i)
        spectra.append([truth, observed, sigma, fit["flux_single"], fit["flux_binary"]])
        fits.append(fit)
    fields = list(dict.fromkeys(k for r in results for k in r))
    with (args.out / "selection.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    np.savez_compressed(args.out / "spectra.npz", wave_nm=model.wave_nm,
                        input_row=np.array(indices), spectra=np.array(spectra))
    supported = [r for r in results if r["status"] == "ok"]
    print(f"Model-supported systems: {len(supported)}/{len(rows)}")
    for name in ["pass_quality", "pass_improvement", "pass_contrast_g", "pass_contrast_rp", "pass_wd_cmd", "pass_selection"]:
        print(f"{name}: {sum(r[name] for r in supported)}/{len(supported)}")
    if not spectra:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    truth, observed, sigma, single, binary = spectra[0]
    axes[0].errorbar(model.wave_nm, observed, yerr=sigma, fmt=".", ms=3, alpha=.5, label="Mock XP")
    axes[0].plot(model.wave_nm, truth, color="black", label="Input binary")
    axes[0].plot(model.wave_nm, single, label="Single fit")
    axes[0].plot(model.wave_nm, binary, label="Binary fit")
    axes[0].set(xlabel="Wavelength (nm)", ylabel="Flux at 10 pc (model units)")
    axes[0].legend(fontsize=8)
    for passed, marker, label in [(False, "x", "Rejected"), (True, "o", "Selected")]:
        group = [r for r in supported if r["pass_selection"] == passed]
        axes[1].scatter([r["bp_rp_wd"] for r in group],
                        [r["abs_g_wd"]-r["abs_g_ms"] for r in group], marker=marker, label=label)
    axes[1].axhline(2.5, color="grey", ls="--")
    axes[1].axhline(-2.5, color="grey", ls="--")
    axes[1].set(xlabel="Input WD BP-RP", ylabel="Input M_G(WD) - M_G(MS)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "diagnostic.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
