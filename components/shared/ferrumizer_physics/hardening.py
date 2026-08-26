"""Hardening stage: JMAK + Scheil additivity, Koistinen-Marburger,
smoothstep hardness mixing, and effective case depth (ECD).

All functions are JAX-compatible and fully differentiable.

Phase accounting (v0.2 hardening model)
--------------------------------------
The quench model tracks FOUR microstructural constituents per depth node:

    f_martensite — austenite that survived the diffusional noses and
                   transformed by Koistinen-Marburger once the local
                   temperature dropped below Ms.
    f_bainite    — lower/upper bainite formed by Scheil-JMAK at the
                   bainite nose (~450 C).
    f_pearlite   — pearlite formed at the pearlite nose (~600 C).
    f_rest       — everything else (untransformed austenite / ferrite),
                   assigned the core hardness.

Hardness is a rule of mixtures over the four fractions with phase-specific
HV values (``H_bainite`` / ``H_pearlite`` in the preset) instead of the
legacy "all non-martensite == Hcore" collapse. This is what makes a
bainitic case look harder than a ferritic core — the C-curve shape a
metallurgist expects from a slow quench.
"""

from __future__ import annotations

import jax.numpy as jnp

# Martensite-finish offset: Mf = Ms - Mf_OFFSET_K. Koistinen-Marburger
# transformation stops at Mf; below it no additional martensite forms.
# Ms - Mf ~ 200 K is the standard engineering approximation for low-alloy
# steels (see e.g. Krauss, "Steels: Processing, Structure, and Performance").
MF_OFFSET_K = 200.0

# Retained-austenite model: above C_RA_START mass-% C the case retains
# austenite, which softens the martensite. f_RA ramps linearly from 0 at
# C_RA_START to C_RA_MAX_FRACTION at C_RA_END. Empirically, >1.0-1.1 % C
# gives noticeable retained austenite in carburized cases (ASM Handbook
# Vol. 4); hardness drop can reach 50-100 HV at 1.3 % C.
C_RA_START = 1.0
C_RA_END = 1.3
C_RA_MAX_FRACTION = 0.35
H_RA_DEFAULT = 300.0  # HV, retained austenite is soft (~300 HV)

# As-quenched martensite hardness vs carbon (ASM Handbook Vol. 4, the
# "hardness of fully martensitic steel" curve; low-C end anchored to the
# measured 8620 Jominy quench-end, 43.6 HRC ~ 430 HV, 2018 instrumented rig).
# Low-carbon martensite is NOT soft: a 0.19 % C Jominy bar at the quenched
# end reads ~430 HV (~43 HRC), and hardness climbs steeply with C to
# ~0.8-1.0 % C before the retained-austenite roll-off.
MARTENSITE_HV_ANCHORS = (
    (0.05, 220.0), (0.10, 300.0), (0.15, 360.0), (0.20, 430.0),
    (0.25, 470.0), (0.30, 505.0), (0.35, 535.0), (0.40, 565.0),
    (0.50, 610.0), (0.60, 655.0), (0.70, 690.0), (0.80, 720.0),
    (0.90, 740.0), (1.00, 755.0),
)


def martensite_hardness(C):
    """Full-martensite hardness (HV) as a function of carbon (mass-%).

    C1-smooth monotone (PCHIP-style, Fritsch-Carlson) interpolation through
    the ASM as-quenched martensite anchors. This replaces the old assumption
    that any carbon below Cmin behaves like Hcore — physically wrong for
    low-carbon martensite (Jominy bars, cores of lean steels), where 0.19 % C
    still hardens to ~400 HV.

    The C1 smoothness is deliberate: the piecewise-linear ``jnp.interp`` has
    a gradient kink at every anchor, and HMC/NUTS (V7's SBC gate) assumes a
    smooth log-density — kinks show up as under-covering posteriors. PCHIP
    keeps the anchor values EXACTLY (the published curve) while making the
    gradient continuous. Clamped to the anchor range (flat beyond it).
    """
    C = jnp.asarray(C, dtype=jnp.float64)
    xs = jnp.asarray([a[0] for a in MARTENSITE_HV_ANCHORS], jnp.float64)
    ys = jnp.asarray([a[1] for a in MARTENSITE_HV_ANCHORS], jnp.float64)
    Cc = jnp.clip(C, xs[0], xs[-1])
    return _pchip(xs, ys, Cc)


def _pchip(xs, ys, x):
    """Fritsch-Carlson monotone cubic Hermite interpolation (C1, no kinks).

    Preserves the anchor values exactly and is monotone between them (no
    overshoot), with continuous first derivatives. This is the standard
    shape-preserving interpolation used for materials data tables.
    """
    h = jnp.diff(xs)
    delta = jnp.diff(ys) / h
    n = len(xs)

    # secant-based slopes (Fritsch-Carlson): harmonic mean of neighbours,
    # forced to zero at local extrema so the interpolant stays monotone.
    # NOTE: jnp.where everywhere — under jit/grad the delta values are
    # traced arrays and Python `if` would raise TracerBoolConversionError.
    def slope_at(i):
        if i == 0:
            return _slope_endpoint(delta[0], delta[1])
        if i == n - 1:
            return _slope_endpoint(delta[-1], delta[-2])
        d_prev, d_next = delta[i - 1], delta[i]
        w_prev, w_next = 2.0 * h[i] + h[i - 1], h[i] + 2.0 * h[i - 1]
        same_sign = d_prev * d_next > 0.0
        hm = (w_prev + w_next) / (w_prev / d_prev + w_next / d_next)
        return jnp.where(same_sign, hm, 0.0)

    m = jnp.stack([slope_at(i) for i in range(n)])

    # locate the bracketing interval; side="right" then -1 puts x exactly on
    # a knot into the LEFT interval (t=0), keeping the gradient well-defined
    # at the knots themselves (a measure-zero boundary artifact otherwise).
    idx = jnp.clip(jnp.searchsorted(xs, x, side="right") - 1, 0, n - 2)
    t = (x - xs[idx]) / h[idx]
    t2, t3 = t * t, t * t * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00 * ys[idx] + h10 * h[idx] * m[idx]
        + h01 * ys[idx + 1] + h11 * h[idx] * m[idx + 1]
    )


def _slope_endpoint(d_first, d_second):
    """Endpoint slope: harmonic mean of the two first secants (FC-77)."""
    same_sign = d_first * d_second > 0.0
    w1, w2 = 2.0 * d_first + d_second, d_first + 2.0 * d_second
    hm = (w1 + w2) / (w1 / d_first + w2 / d_second)
    return jnp.where(same_sign, hm, 0.0)


def ms_andrews(C_wt_pct, A: float, b_carbon: float):
    """Andrews (1965) martensite-start temperature in K.

    Ms(C) = A - b_carbon * C,  C in mass-%.
    """
    return A - b_carbon * jnp.asarray(C_wt_pct, dtype=jnp.float64)


def km_fraction(Ms, Tq: float, alpha_km: float, mf_offset_K: float = MF_OFFSET_K):
    """Koistinen-Marburger martensite volume fraction with Mf cutoff.

    f_M = 1 - exp(-alpha_km * dT),  dT saturates smoothly at Ms - Mf.

    The Mf cutoff fixes the physical flaw of evaluating KM at the final
    bath temperature with no floor: for a 120 C bath and a high-carbon
    case where Mf ~ 100 C, the old model kept predicting martensite all
    the way to the bath. With the floor, dT stops at Ms - Mf, so the
    fraction saturates at its Mf value and does not keep growing for
    colder baths. (Integrating KM kinetically along the cooling curve is
    the next refinement; the floor is the honest one-line fix.)

    The saturation uses a SMOOTH soft-plus cap instead of a hard
    ``min``: the hard clamp has a gradient kink at dT = Ms - Mf, and
    HMC/NUTS (which V7's SBC gate runs on) assumes a smooth log-density —
    a kink there makes the leapfrog integrator inaccurate and the adapted
    step size wrong, which shows up as under-covering posteriors in SBC.
    Physically this is also the better statement: Mf is the ~99% completion
    temperature, not a hard stop, so the fraction should asymptote.
    """
    Ms = jnp.asarray(Ms, dtype=jnp.float64)
    dT_raw = jnp.maximum(Ms - Tq, 0.0)
    # Smooth approximation of min(dT_raw, mf_offset) with NO gradient kink:
    #   dT = dT_raw - softplus(dT_raw - mf_offset)
    # which has slope 1 near 0 (preserving the calibrated linear KM regime
    # just below Ms), asymptotes to exactly mf_offset as dT_raw -> inf
    # (same saturation value as the hard clamp), and is C^inf smooth.
    # HMC/NUTS (which V7's SBC gate runs on) assumes a smooth log-density;
    # the hard clamp's kink at dT = Ms - Mf made the leapfrog integrator
    # inaccurate and showed up as under-covering posteriors in SBC.
    dT = dT_raw - jnp.log1p(jnp.exp(dT_raw - mf_offset_K))
    return 1.0 - jnp.exp(-alpha_km * dT)


def smoothstep(u):
    """C^1 smoothstep  3u^2 - 2u^3  with a soft clamp to [0, 1].

    Note: the soft clamp is still a clamp — u is clipped so the mixing
    weight lives in [0, 1]; smoothstep only makes the transition C^1
    continuous. (Review: don't oversell "never a hard clamp".)
    """
    uc = jnp.clip(jnp.asarray(u, dtype=jnp.float64), 0.0, 1.0)
    return 3.0 * uc**2 - 2.0 * uc**3


def retained_austenite_fraction(C):
    """Retained-austenite fraction vs case carbon (mass-%).

    Ramp: 0 below C_RA_START, linear to C_RA_MAX_FRACTION at C_RA_END.
    This captures the real drop in as-quenched hardness for hypereutectoid
    cases (1.0+ % C) that the old C >= Cideal -> Hmax clamp missed.
    """
    C = jnp.asarray(C, dtype=jnp.float64)
    u = (C - C_RA_START) / (C_RA_END - C_RA_START)
    return C_RA_MAX_FRACTION * smoothstep(u)


def hardness_profile(C, preset: dict, f_mart=None):
    """Hardness (HV) of the MARTENSITE fraction vs carbon, with RA roll-off.

    H_mart(C) = ASM as-quenched martensite curve (raw, not renormalized to
    Hcore): a 0.19 % C Jominy bar at full martensite is ~400 HV (~41 HRC),
    NOT Hcore. The preset's Hcore only prices the NON-martensite rest
    (ferrite/pearlite/retained austenite) via the rule of mixtures in
    :func:`mix_phase_hardness`.

    Above C_RA_START the retained-austenite fraction ramps up and the
    martensite hardness is diluted toward H_RA:

        H_mart = (1 - f_RA) * H_plateau + f_RA * H_RA

    Optionally weighted by martensite fraction (rule of mixtures).
    """
    h = preset["hardness"]
    C = jnp.asarray(C, dtype=jnp.float64)

    H_plateau = martensite_hardness(C)
    # retained austenite dilutes the martensite hardness
    H_RA = h.get("H_RA", H_RA_DEFAULT)
    f_RA = retained_austenite_fraction(C)
    H_mart = (1.0 - f_RA) * H_plateau + f_RA * H_RA
    if f_mart is not None:
        return f_mart * H_mart + (1.0 - f_mart) * h["Hcore"]
    return H_mart


def mix_phase_hardness(f_mart, f_bainite, f_pearlite, preset, H_mart, C=None):
    """Rule of mixtures with phase-specific hardness (the honest mixing rule).

    H = f_mart*H_mart + f_bain*H_bainite + f_pearl*H_pearl(C) + f_rest*Hcore

    f_rest is whatever austenite did not transform (retained austenite /
    ferrite) and gets the core hardness. Without this split, bainite was
    silently priced at Hcore (230 HV for 8620) — but lower bainite in a
    carburized case runs 350-400 HV. A slow-quench bainitic case now shows
    up as hard, not as a flat "core" line.

    ``C`` (optional, mass-%) prices the pearlite by the local carbon via the
    eutectoid lever rule: at core carbon (~0.2 %) the diffusional product is
    mostly ferrite (hardness ~ Hcore), at eutectoid carbon (0.77 %) and
    above it is full pearlite (H_pearlite). This is what makes the Jominy
    far end (~0.2 % C, fully pearlitic) soft (~20 HRC) instead of
    mis-pricing it at the full-pearlite hardness (~27 HRC).
    """
    h = preset["hardness"]
    H_bain = h.get("H_bainite", h["Hcore"] + 150.0)
    H_pearl_full = h.get("H_pearlite", h["Hcore"] + 40.0)
    if C is not None:
        # eutectoid lever rule: pearlite fraction of the diffusional product
        C = jnp.asarray(C, dtype=jnp.float64)
        C_core = float(preset.get("C0", 0.2))
        pearl_frac = jnp.clip((C - C_core) / (0.77 - C_core), 0.0, 1.0)
        H_pearl = H_pearl_full * pearl_frac + h["Hcore"] * (1.0 - pearl_frac)
    else:
        H_pearl = H_pearl_full
    f_rest = 1.0 - f_mart - f_bainite - f_pearlite
    H = (
        f_mart * H_mart
        + f_bainite * H_bain
        + f_pearlite * H_pearl
        + f_rest * h["Hcore"]
    )
    return H


def ecd_from_hardness(H, x_mm, threshold: float = 550.0):
    """Effective case depth (mm): depth at which H crosses *threshold*.

    Fully differentiable: for each segment [i, i+1] the fraction of the
    segment that lies above the threshold is computed via clamped linear
    interpolation, then summed.  For a monotonically decreasing profile this
    reproduces the exact ISO 2639 crossing depth.

    Segment logic (robust to non-monotone and flat profiles):
      * both endpoints above threshold -> entire segment counts (frac 1)
      * both endpoints below threshold -> nothing counts (frac 0)
      * straddling -> linear-interpolation crossing fraction
    The old formula only clipped ``(H_left - thr)/(H_left - H_right)`` to
    [0, 1], which mis-scored flat near-threshold segments whose tiny
    negative denominator exploded to +1 — producing phantom case depth
    (e.g. 7 mm ECD on a part with no node above 550 HV).
    """
    H = jnp.asarray(H, dtype=jnp.float64)
    x_mm = jnp.asarray(x_mm, dtype=jnp.float64)

    H_left = H[:-1]
    H_right = H[1:]
    dx_seg = x_mm[1:] - x_mm[:-1]

    denom = H_left - H_right
    safe_denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
    frac = (H_left - threshold) / safe_denom

    both_above = (H_left >= threshold) & (H_right >= threshold)
    both_below = (H_left < threshold) & (H_right < threshold)
    frac = jnp.where(both_above, 1.0, jnp.where(both_below, 0.0, frac))
    frac = jnp.clip(frac, 0.0, 1.0)

    return jnp.sum(frac * dx_seg)


def jmak_scheil_fraction(T_history, dt, n_exp, k_ref, T_nose, width):
    """Scheil-additivity JMAK transformed fraction for diffusional phases.

    k(T) = k_ref * exp(-((T - T_nose)/width)^2)   (Gaussian C-curve approx.)
    X    = 1 - exp( -( sum k(T_i) dt )^n )

    The Gaussian C-curve is a mathematical surrogate for the TTT nose
    (driving force vs diffusion-limited asymmetry are collapsed into one
    symmetric bell). It is a legitimate engineering approximation for an
    emulator, NOT fundamental nucleation-and-growth kinetics — documented
    as such (review: don't oversell the C-curve).
    """
    T = jnp.asarray(T_history, dtype=jnp.float64)
    k_T = k_ref * jnp.exp(-(((T - T_nose) / width) ** 2))
    integral = jnp.sum(k_T) * dt
    return 1.0 - jnp.exp(-(integral**n_exp))


def ttt_start_times(preset: dict, X: float = 0.01, T_grid=None):
    """Isothermal transformation start times (s) for the two C-curves.

    Inverts the JMAK relation at a target fraction X:
        t_start(T) = ([-ln(1 - X)]^(1/n)) / k(T)
    for pearlite and bainite. Plotted as T vs log10(t) this is the classic
    TTT C-curve pair; cooling curves from the spatial quench overlaid on
    the same axes give the CCT-style answer a metallurgist expects
    (review 2: "where is my C-curve?").

    ``T_grid`` (K) defaults to a 200-900 C sweep. Returns dict with
    pearlite/bainite start arrays and the T grid.
    """
    jmak = preset.get("jmak", {})
    n_exp = jmak.get("n", 2.0)
    k_pearlite = jmak.get("k_pearlite", 8.5e-9)
    k_bainite = jmak.get("k_bainite", 1.8e-10)
    if T_grid is None:
        T_grid = jnp.linspace(200.0 + 273.15, 900.0 + 273.15, 200)
    T = jnp.asarray(T_grid, dtype=jnp.float64)
    ln_term = jnp.log(1.0 / (1.0 - X))
    factor = ln_term ** (1.0 / n_exp)

    k_p = k_pearlite * jnp.exp(-(((T - PEARLITE_NOSE_K) / PEARLITE_WIDTH_K) ** 2))
    k_b = k_bainite * jnp.exp(-(((T - BAINITE_NOSE_K) / BAINITE_WIDTH_K) ** 2))
    t_p = factor / k_p
    t_b = factor / k_b
    return {"T": T, "t_pearlite_s": t_p, "t_bainite_s": t_b, "X": X}


# --------------------------------------------------------------------------- #
# Quench model: Newton cooling + Scheil-JMAK diffusional phases
# --------------------------------------------------------------------------- #
# Real quenches are NOT instantaneous: the cooling rate is set by the quench
# medium (oil/water/polymer/air), its temperature, agitation, and part size.
# Slow cooling lets diffusional phases (pearlite/bainite) form, consuming
# austenite that can no longer become martensite — the direct cause of soft
# spots and out-of-spec case depth in production. This model computes a
# lumped Newton cooling curve and integrates Scheil-JMAK C-curves over it.
QUENCH_MEDIA_H = {
    "air": 50.0,        # W/m^2/K  — still air, slow
    "oil": 900.0,       # W/m^2/K  — typical quench oil
    "polymer": 1800.0,  # W/m^2/K  — polymer quenchant (PVP type)
    "water": 3500.0,    # W/m^2/K  — agitated water
}
# Medium-specific C-curve noses (K) for the two diffusional phases
# (pearlite ~600 C, bainite ~450 C for low-alloy carburizing steels).
PEARLITE_NOSE_K = 873.15
BAINITE_NOSE_K = 723.15
PEARLITE_WIDTH_K = 90.0
BAINITE_WIDTH_K = 70.0


def newton_cooling_curve(
    T_start,
    T_quench,
    h_quench,
    rho_cp,
    half_thickness_m,
    t_samples,
    agitation: float = 0.5,
    geometry: str = "slab",
):
    """Lumped-Newton cooling T(t) = Tq + (T0 - Tq) exp(-t / tau).

    tau = rho*cp*L_char / (h*(1 + agitation))

    Characteristic length L_char is geometry-aware:
      * slab    -> half-thickness (surface-to-center distance)
      * cylinder -> R/2 (volume/area = (pi R^2)/(2 pi R) = R/2)
    The old model used half_thickness_m for cylinders too, overestimating
    the thermal mass by 2x and slowing the modeled cylinder quench by 2x.
    Agitation scales the effective film coefficient. Fully differentiable
    in all inputs.
    """
    L = half_thickness_m
    if geometry == "cylinder":
        L = half_thickness_m / 2.0
    tau = (rho_cp * L) / (h_quench * (1.0 + agitation))
    T = T_quench + (T_start - T_quench) * jnp.exp(-jnp.asarray(t_samples, jnp.float64) / tau)
    return T


def quench_fractions(
    C_profile,
    Ms_profile,
    preset: dict,
    T_quench,
    T_start,
    h_quench,
    rho_cp,
    half_thickness_m,
    agitation: float = 0.5,
    t_quench_total: float = 600.0,
    n_samples: int = 120,
    geometry: str = "slab",
):
    """Martensite / diffusional fractions for a finite-rate quench.

    Returns dict with f_martensite (per-depth), f_diffusional (per-depth,
    pearlite + bainite via sequential Scheil-JMAK), the cooling curve, and
    the resulting hardness profile (phase-specific rule of mixtures).

    Physics: austenite first transforms to pearlite/bainite while the part
    cools through the C-curve noses; the austenite that survives to Ms
    becomes martensite (Koistinen-Marburger). A slow quench (thick part,
    mild oil, no agitation) consumes most austenite as diffusional phases,
    so little martensite forms and the hardness/ECD collapses — exactly the
    production failure mode this model exists to predict.
    """
    C = jnp.asarray(C_profile, jnp.float64)
    Ms = jnp.asarray(Ms_profile, jnp.float64)
    t = jnp.linspace(0.0, t_quench_total, n_samples)
    cooling = newton_cooling_curve(
        T_start, T_quench, h_quench, rho_cp, half_thickness_m, t, agitation, geometry
    )
    dt = t_quench_total / n_samples

    jmak = preset.get("jmak", {})
    n_exp = jmak.get("n", 2.0)
    k_pearlite = jmak.get("k_pearlite", 8.5e-9)
    k_bainite = jmak.get("k_bainite", 1.8e-10)

    X_pearlite = jmak_scheil_fraction(
        cooling, dt, n_exp, k_pearlite, PEARLITE_NOSE_K, PEARLITE_WIDTH_K
    )
    # bainite can only form from austenite not already consumed by pearlite
    X_bainite = (1.0 - X_pearlite) * jmak_scheil_fraction(
        cooling, dt, n_exp, k_bainite, BAINITE_NOSE_K, BAINITE_WIDTH_K
    )
    X_diff = X_pearlite + X_bainite

    # surviving austenite -> martensite (Mf cutoff applied inside KM)
    f_mart = (1.0 - X_diff) * km_fraction(Ms, T_quench, preset["km_alpha"],
                                 preset.get("mf_offset_K", MF_OFFSET_K))

    # phase-specific rule of mixtures — bainite/pearlite priced at their
    # own hardness, not collapsed into Hcore
    h = preset["hardness"]
    H_mart = hardness_profile(C, preset, None)
    H = mix_phase_hardness(f_mart, X_bainite, X_pearlite, preset, H_mart, C)

    return {
        "cooling_curve": cooling,
        "X_pearlite": X_pearlite,
        "X_bainite": X_bainite,
        "X_diffusional": X_diff,
        "f_martensite": f_mart,
        "H": H,
    }


def quench_fractions_depth(
    C_profile,
    Ms_profile,
    preset: dict,
    T_history_matrix,  # (M, n) K, rows = time, cols = depth (surface->core)
    dt: float,
    T_quench,
):
    """Per-depth phase fractions from a SPATIAL cooling history (CCT-style).

    ``T_history_matrix`` comes from :func:`thermal.run_quench_thermal`: each
    column is the cooling curve at one depth node. Because cooling rate now
    varies through the section, bainite/pearlite formation differs by depth —
    the surface cools fast and keeps martensite; the core cools slow and can
    form diffusional phases. This is the answer to "show me the phase
    fractions across the section," which a lumped single-curve model cannot
    give.

    Hardness uses the phase-specific rule of mixtures (bainite != pearlite
    != core), so a slow-quench part shows a hard bainitic case over a soft
    pearlitic core instead of a flat scaled line.

    Returns per-depth X_pearlite, X_bainite, X_diffusional, f_martensite,
    and the resulting hardness profile.
    """
    T = jnp.asarray(T_history_matrix, jnp.float64)
    Ms = jnp.asarray(Ms_profile, jnp.float64)
    C = jnp.asarray(C_profile, jnp.float64)
    jmak = preset.get("jmak", {})
    n_exp = jmak.get("n", 2.0)
    k_pearlite = jmak.get("k_pearlite", 8.5e-9)
    k_bainite = jmak.get("k_bainite", 1.8e-10)

    # Scheil-JMAK integral per depth column. Vectorized across depths:
    # k_T shape (M, n); integral over time axis per column.
    k_pearl_T = k_pearlite * jnp.exp(-(((T - PEARLITE_NOSE_K) / PEARLITE_WIDTH_K) ** 2))
    k_bain_T = k_bainite * jnp.exp(-(((T - BAINITE_NOSE_K) / BAINITE_WIDTH_K) ** 2))
    int_pearl = jnp.sum(k_pearl_T, axis=0) * dt
    int_bain = jnp.sum(k_bain_T, axis=0) * dt
    X_pearlite = 1.0 - jnp.exp(-(int_pearl**n_exp))
    X_bainite_raw = 1.0 - jnp.exp(-(int_bain**n_exp))
    X_bainite = (1.0 - X_pearlite) * X_bainite_raw
    X_diff = X_pearlite + X_bainite

    f_mart = (1.0 - X_diff) * km_fraction(Ms, T_quench, preset["km_alpha"],
                                 preset.get("mf_offset_K", MF_OFFSET_K))

    h = preset["hardness"]
    H_mart = hardness_profile(C, preset, None)
    H = mix_phase_hardness(f_mart, X_bainite, X_pearlite, preset, H_mart, C)

    return {
        "X_pearlite": X_pearlite,
        "X_bainite": X_bainite,
        "X_diffusional": X_diff,
        "f_martensite": f_mart,
        "H": H,
    }


def run_hardening(
    C_profile, x_mm, T_quench: float, preset: dict, T_history=None, dt: float = 1.0
) -> dict:
    """Full hardening stage: KM martensite + hardness + ECD.

    Parameters
    ----------
    C_profile : (n,) carbon mass-%, surface -> core
    x_mm      : (n,) depth from surface in mm
    T_quench  : quench / room temperature (K)
    preset    : alloy preset dict
    T_history : optional cooling curve (K) for JMAK diffusional fraction
    dt        : time-step of T_history (s)

    Returns dict with Ms, f_martensite, H, ecd_mm (+ optionally X_diff).
    """
    C_profile = jnp.asarray(C_profile, dtype=jnp.float64)
    x_mm = jnp.asarray(x_mm, dtype=jnp.float64)

    ms_cfg = preset["ms"]
    Ms = ms_andrews(C_profile, ms_cfg["A"], ms_cfg["b_carbon"])
    f_mart = km_fraction(Ms, T_quench, preset["km_alpha"], preset.get("mf_offset_K", 200.0))
    H = hardness_profile(C_profile, preset, f_mart)
    ecd = ecd_from_hardness(H, x_mm, preset["ecd_threshold_hv"])

    result = {
        "Ms": Ms,
        "f_martensite": f_mart,
        "H": H,
        "ecd_mm": ecd,
    }

    if T_history is not None:
        jmak = preset.get("jmak", {})
        X_diff = jmak_scheil_fraction(
            T_history,
            dt,
            n_exp=jmak.get("n", 2.0),
            k_ref=jmak.get("k_pearlite", 8.5e-9),
            T_nose=jmak.get("T_nose", 823.15),
            width=jmak.get("width", 80.0),
        )
        result["X_diffusional"] = X_diff

    return result
