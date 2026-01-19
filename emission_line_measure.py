#!/usr/bin/env python3
"""Measure emission-line fluxes and continuum from 1D spectra.

Assumes input spectrum is a whitespace-delimited text file readable by
numpy.loadtxt with columns:
  wavelength[A], flux[erg/s/cm^2/A], flux_error[erg/s/cm^2/A] (optional)

The script refines redshift around an initial guess, fits a broken power-law
continuum in the rest frame, and integrates line fluxes in velocity windows.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

C_KMS = 299792.458

LINE_LIST = {
    "OIII_5007": 5006.843,
    "OIII_4959": 4958.911,
    "Hb": 4861.333,
    "Ha": 6562.819,
    "OIII_4363": 4363.209,
    "CIV_1549": 1549.480,
    "NIV_1486": 1486.500,
    "HeII_1640": 1640.420,
    "CIII_1909": 1908.734,
}


@dataclasses.dataclass
class Spectrum:
    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray

    def rest_wavelength(self, z: float) -> np.ndarray:
        return self.wavelength / (1.0 + z)


@dataclasses.dataclass
class ContinuumFit:
    break_wave: float
    alpha_blue: float
    alpha_red: float
    norm: float

    def model(self, rest_wave: np.ndarray) -> np.ndarray:
        scaled = rest_wave / self.break_wave
        blue = scaled ** self.alpha_blue
        red = scaled ** self.alpha_red
        return self.norm * np.where(rest_wave <= self.break_wave, blue, red)


def load_spectrum(path: str) -> Spectrum:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Input file must have at least wavelength and flux columns.")
    wavelength = data[:, 0]
    flux = data[:, 1]
    if data.shape[1] >= 3:
        error = data[:, 2]
    else:
        error = np.full_like(flux, np.nan)

    mask = np.isfinite(wavelength) & np.isfinite(flux)
    wavelength = wavelength[mask]
    flux = flux[mask]
    error = error[mask]

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    flux = flux[order]
    error = error[order]

    if not np.all(np.isfinite(error)):
        median = np.nanmedian(flux)
        scatter = np.nanmedian(np.abs(flux - median)) * 1.4826
        if not np.isfinite(scatter) or scatter <= 0:
            scatter = np.nanstd(flux)
        error = np.full_like(flux, scatter if scatter > 0 else 1.0)

    return Spectrum(wavelength=wavelength, flux=flux, error=error)


def mask_lines(rest_wave: np.ndarray, z: float, velocity_kms: float) -> np.ndarray:
    mask = np.ones_like(rest_wave, dtype=bool)
    for rest in LINE_LIST.values():
        delta = rest * (velocity_kms / C_KMS)
        mask &= (rest_wave < rest - delta) | (rest_wave > rest + delta)
    return mask


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        raise ValueError("Not enough points for linear fit.")
    wsum = np.sum(w)
    xw = np.sum(w * x) / wsum
    yw = np.sum(w * y) / wsum
    cov = np.sum(w * (x - xw) * (y - yw))
    var = np.sum(w * (x - xw) ** 2)
    slope = cov / var
    intercept = yw - slope * xw
    return intercept, slope


def fit_continuum(
    spectrum: Spectrum,
    z: float,
    break_wave: float = 3646.0,
    line_mask_velocity: float = 1500.0,
) -> ContinuumFit:
    rest_wave = spectrum.rest_wavelength(z)
    mask = mask_lines(rest_wave, z, line_mask_velocity)
    mask &= np.isfinite(spectrum.flux) & (spectrum.flux > 0)
    masked_wave = rest_wave[mask]
    masked_flux = spectrum.flux[mask]
    masked_error = spectrum.error[mask]

    if masked_wave.size < 10:
        raise ValueError("Not enough continuum points to fit.")

    blue = masked_wave <= break_wave
    red = masked_wave > break_wave

    if blue.sum() < 3 or red.sum() < 3:
        log_wave = np.log10(masked_wave)
        log_flux = np.log10(masked_flux)
        weights = 1.0 / np.clip(masked_error, 1e-30, None)
        intercept, slope = weighted_linear_fit(log_wave, log_flux, weights)
        norm = 10 ** intercept
        return ContinuumFit(break_wave=break_wave, alpha_blue=slope, alpha_red=slope, norm=norm)

    def fit_segment(segment: np.ndarray) -> Tuple[float, float]:
        log_wave = np.log10(masked_wave[segment])
        log_flux = np.log10(masked_flux[segment])
        weights = 1.0 / np.clip(masked_error[segment], 1e-30, None)
        return weighted_linear_fit(log_wave, log_flux, weights)

    intercept_blue, slope_blue = fit_segment(blue)
    intercept_red, slope_red = fit_segment(red)

    norm = 10 ** intercept_red * (break_wave ** -slope_red)
    norm_blue = 10 ** intercept_blue * (break_wave ** -slope_blue)
    norm = 0.5 * (norm + norm_blue)

    return ContinuumFit(
        break_wave=break_wave,
        alpha_blue=slope_blue,
        alpha_red=slope_red,
        norm=norm,
    )


def integrate_line(
    spectrum: Spectrum,
    z: float,
    rest_wave: float,
    continuum: ContinuumFit,
    velocity_window: float,
) -> Tuple[float, float, float]:
    obs_center = rest_wave * (1.0 + z)
    delta = obs_center * (velocity_window / C_KMS)
    mask = (spectrum.wavelength >= obs_center - delta) & (
        spectrum.wavelength <= obs_center + delta
    )
    if not np.any(mask):
        return math.nan, math.nan, math.nan

    wave = spectrum.wavelength[mask]
    flux = spectrum.flux[mask]
    err = spectrum.error[mask]

    rest_wave_segment = wave / (1.0 + z)
    cont = continuum.model(rest_wave_segment)
    flux_sub = flux - cont

    line_flux = np.trapz(flux_sub, wave)
    dw = np.gradient(wave)
    line_err = np.sqrt(np.sum((err * dw) ** 2))
    snr = line_flux / line_err if line_err > 0 else math.nan
    return line_flux, line_err, snr


def continuum_at_rest(
    continuum: ContinuumFit,
    rest_wave: float,
) -> float:
    return float(continuum.model(np.array([rest_wave]))[0])


def flux_to_abmag(flux_lambda: float, rest_wave: float) -> float:
    if flux_lambda <= 0:
        return math.nan
    c = 2.99792458e18  # A/s
    flux_nu = flux_lambda * rest_wave**2 / c
    return -2.5 * math.log10(flux_nu) - 48.6


def score_redshift(
    spectrum: Spectrum,
    z: float,
    velocity_window: float,
) -> float:
    score = 0.0
    for rest in LINE_LIST.values():
        obs_center = rest * (1.0 + z)
        delta = obs_center * (velocity_window / C_KMS)
        mask = (spectrum.wavelength >= obs_center - delta) & (
            spectrum.wavelength <= obs_center + delta
        )
        if mask.sum() < 3:
            continue
        wave = spectrum.wavelength[mask]
        flux = spectrum.flux[mask]
        err = spectrum.error[mask]
        median = np.nanmedian(flux)
        flux_sub = flux - median
        line_flux = np.trapz(flux_sub, wave)
        dw = np.gradient(wave)
        line_err = np.sqrt(np.sum((err * dw) ** 2))
        if line_err <= 0:
            continue
        snr = line_flux / line_err
        if snr > 0:
            score += snr
    return score


def refine_redshift(
    spectrum: Spectrum,
    z_guess: float,
    z_range: float,
    z_step: float,
    velocity_window: float,
) -> float:
    z_grid = np.arange(z_guess - z_range, z_guess + z_range + z_step, z_step)
    scores = np.array([
        score_redshift(spectrum, z, velocity_window) for z in z_grid
    ])
    best = np.nanargmax(scores)
    return float(z_grid[best])


def summarize_lines(
    spectrum: Spectrum,
    z: float,
    continuum: ContinuumFit,
    velocity_window: float,
    snr_threshold: float,
) -> List[Dict[str, float]]:
    results = []
    for name, rest in LINE_LIST.items():
        line_flux, line_err, snr = integrate_line(
            spectrum, z, rest, continuum, velocity_window
        )
        detected = snr >= snr_threshold if np.isfinite(snr) else False
        results.append(
            {
                "line": name,
                "rest_wave": rest,
                "flux": line_flux,
                "flux_err": line_err,
                "snr": snr,
                "detected": float(detected),
            }
        )
    return results


def format_results(results: Iterable[Dict[str, float]]) -> str:
    header = "line rest_wave flux flux_err snr detected"
    lines = [header]
    for row in results:
        lines.append(
            "{line} {rest_wave:.2f} {flux:.4e} {flux_err:.4e} {snr:.2f} {detected:.0f}".format(
                **row
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", help="Path to input spectrum text file")
    parser.add_argument("--z", type=float, required=True, help="Initial redshift guess")
    parser.add_argument("--z-range", type=float, default=0.02, help="Search range around z")
    parser.add_argument("--z-step", type=float, default=2e-4, help="Redshift step")
    parser.add_argument(
        "--velocity-window",
        type=float,
        default=1200.0,
        help="Half-window for integration in km/s",
    )
    parser.add_argument(
        "--line-mask-velocity",
        type=float,
        default=1500.0,
        help="Mask width for continuum fit in km/s",
    )
    parser.add_argument(
        "--break-wave",
        type=float,
        default=3646.0,
        help="Rest-frame break wavelength in Angstrom",
    )
    parser.add_argument(
        "--uv-rest-wave",
        type=float,
        default=1500.0,
        help="Rest wavelength for UV magnitude",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=3.0,
        help="SNR threshold to flag detections",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file for line measurements",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spectrum = load_spectrum(args.spectrum)
    z_refined = refine_redshift(
        spectrum,
        args.z,
        args.z_range,
        args.z_step,
        args.velocity_window,
    )
    continuum = fit_continuum(
        spectrum,
        z_refined,
        break_wave=args.break_wave,
        line_mask_velocity=args.line_mask_velocity,
    )
    results = summarize_lines(
        spectrum,
        z_refined,
        continuum,
        args.velocity_window,
        args.snr_threshold,
    )
    uv_flux = continuum_at_rest(continuum, args.uv_rest_wave)
    uv_mag = flux_to_abmag(uv_flux, args.uv_rest_wave)

    summary = [
        f"Refined z: {z_refined:.6f}",
        f"Continuum break {continuum.break_wave:.1f}A",
        f"alpha_blue={continuum.alpha_blue:.3f} alpha_red={continuum.alpha_red:.3f}",
        f"UV continuum @ {args.uv_rest_wave:.1f}A: {uv_flux:.4e}",
        f"UV AB magnitude: {uv_mag:.3f}",
        "",
        format_results(results),
    ]
    output_text = "\n".join(summary)
    print(output_text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output_text + "\n")


if __name__ == "__main__":
    main()
