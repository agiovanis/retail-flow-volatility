"""
================================================================================
EXPERIMENT CONFIGURATION  –  Retail Flow as a Hidden Risk Factor
Edit ONE block and run the script. Each run produces a single clean output
(Excel + PDF + plots) for that variant.
================================================================================

HOW TO USE
----------
1. Pick the ACTIVE_EXPERIMENT you want to run (built from the dicts below).
2. Set ACTIVE_EXPERIMENT = '<id>'
3. Run:  python retail_flow_engine.py
4. Output lands in OUTPUT_ROOT/<EXPERIMENT_ID>/

NO simultaneous loops across all combinations – one config -> one clean result.

LOOKAHEAD PROTECTION (unchanged from the original 4-file version)
-------------------------------------------------------------------
- Feature row at date t uses only data dated <= t.
- target_rv_h at row t is realized volatility computed strictly from returns
  t+1 ... t+h — never from data at or before t.
- The reflex beta (historical buy-the-dip slope) at t is estimated only from
  (return_s, retail-response_{s+1}) pairs that were fully observed BEFORE day
  t — the pair for day t itself is never inside its own training window.
- Walk-forward refits use returns.iloc[:start] (strictly before the test
  block being predicted) — no shuffling, no k-fold leakage across time.
- assert_no_lookahead() re-derives the target independently for a sample of
  rows and raises if the values disagree; the engine runs it before every
  backtest.
================================================================================
"""

import os

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
OUTPUT_ROOT = r"C:\Users\Diony\Downloads\results_retail_flow"
# For local testing outside Windows, override via env var instead of editing this line:
#   RETAIL_FLOW_OUTPUT_ROOT=/tmp/results_retail_flow python retail_flow_engine.py
OUTPUT_ROOT = os.environ.get("RETAIL_FLOW_OUTPUT_ROOT", OUTPUT_ROOT)


# ---------------------------------------------------------------------------
# DATA SOURCES
# ---------------------------------------------------------------------------
DATA_SOURCES = {
    # ── 1. Real data (primary result) ─────────────────────────────────────
    "real": {
        "label": "Real data — FINRA off-exchange proxy + SPY/VIX (yfinance)",
        "demo": False,
        "note": "Needs outbound access to cdn.finra.org and Yahoo Finance.",
    },

    # ── 2. Synthetic demo (pipeline validation only) ──────────────────────
    "demo": {
        "label": "Synthetic demo data (pipeline validation only, no internet)",
        "demo": True,
        "note": "Ground-truth reflex is baked in — NOT evidence for the paper.",
    },
}

# ---------------------------------------------------------------------------
# TIME PERIODS
# ---------------------------------------------------------------------------
PERIODS = {
    "full":     {"start": "2018-01-01", "end": "2024-12-31",
                 "label": "Full sample (2018–2024)"},
    "post2020": {"start": "2020-01-01", "end": "2024-12-31",
                 "label": "Post-2020 regime (2020–2024)"},
    "covid":    {"start": "2019-06-01", "end": "2021-06-30",
                 "label": "COVID crisis window (2019–2021)"},
    "recent":   {"start": "2022-01-01", "end": "2024-12-31",
                 "label": "Recent regime (2022–2024)"},
}

# ---------------------------------------------------------------------------
# FORECAST HORIZON SETS  (trading days ahead of each feature row)
# ---------------------------------------------------------------------------
HORIZON_SETS = {
    "short":    {"horizons": (5, 10),      "label": "Short horizons (5, 10d)"},
    "standard": {"horizons": (5, 10, 20),  "label": "Standard horizons (5, 10, 20d)"},
    "long":     {"horizons": (10, 20, 40), "label": "Longer horizons (10, 20, 40d)"},
}

# ---------------------------------------------------------------------------
# WALK-FORWARD SETTINGS  (initial training window + refit step, trading days)
# ---------------------------------------------------------------------------
#  NOTE: initial_train must leave room for at least one out-of-sample block
#  AFTER the feature/reflex warm-up is dropped. As a rough guide (standard
#  60d reflex window): "full" and "post2020" periods have ~1700-1200 usable
#  rows (all wfo settings fit); "recent" has ~700 rows ("slow" is too tight);
#  "covid" has only ~450-460 rows (only "fast" fits — "standard"/"slow" leave
#  ZERO out-of-sample rows and every metric would silently come back NaN).
#  The engine now checks this before running and raises a clear error instead
#  of quietly producing an all-NaN result, but picking a compatible pair here
#  saves you the round-trip.
WFO_SETTINGS = {
    "fast":     {"initial_train": 250, "step": 10,
                 "label": "Fast refit (250d initial train, 10d step)"},
    "standard": {"initial_train": 500, "step": 21,
                 "label": "Standard (500d initial train, ~monthly refit)"},
    "slow":     {"initial_train": 750, "step": 63,
                 "label": "Slow refit (750d initial train, ~quarterly)"},
}

# ---------------------------------------------------------------------------
# REFLEX / Z-SCORE ROLLING WINDOW  (trading days)
# ---------------------------------------------------------------------------
REFLEX_WINDOWS = {
    "w40": {"reflex_window": 40, "zscore_window": 40, "label": "40d reflex window"},
    "w60": {"reflex_window": 60, "zscore_window": 60, "label": "60d reflex window (baseline)"},
    "w90": {"reflex_window": 90, "zscore_window": 90, "label": "90d reflex window"},
}

# ---------------------------------------------------------------------------
# RIDGE REGULARIZATION STRENGTH
# ---------------------------------------------------------------------------
RIDGE_ALPHAS = {
    "low":      {"alpha": 0.1, "label": "Light regularization (alpha=0.1)"},
    "standard": {"alpha": 1.0, "label": "Standard regularization (alpha=1.0)"},
    "high":     {"alpha": 5.0, "label": "Heavy regularization (alpha=5.0)"},
}

# ---------------------------------------------------------------------------
# ACTIVE EXPERIMENT  ← THE ONLY THING YOU CHANGE BETWEEN RUNS
# ---------------------------------------------------------------------------
#
#  Format: "<data_source>__<period>__<horizons>__<wfo>__<reflex>__<alpha>"
#
#  Examples:
#    "demo__full__standard__standard__w60__standard"   (safe pipeline check)
#    "real__full__standard__standard__w60__standard"   (primary result)
#    "real__post2020__short__fast__w40__low"           (recent-regime stress test)
#    "real__covid__long__slow__w90__high"              (crisis isolation)
#
# Change wfo from 'standard' (500d initial train) to 'fast' (250d initial train)
ACTIVE_EXPERIMENT = "real__full__standard__fast__w60__high"
# ---------------------------------------------------------------------------
# PARSER – do not edit
# ---------------------------------------------------------------------------

def parse_active():
    """
    Parse ACTIVE_EXPERIMENT string into a ready-to-use config dict.
    Raises ValueError with a helpful message if any key is invalid.
    """
    parts = ACTIVE_EXPERIMENT.strip().split("__")
    if len(parts) != 6:
        raise ValueError(
            f"ACTIVE_EXPERIMENT must have 6 parts separated by '__'.\n"
            f"Got: '{ACTIVE_EXPERIMENT}'\n"
            f"Example: 'demo__full__standard__standard__w60__standard'"
        )

    data_key, period_key, horizon_key, wfo_key, reflex_key, alpha_key = parts

    def _check(key, mapping, name):
        if key not in mapping:
            raise ValueError(
                f"Unknown {name}: '{key}'\n"
                f"Valid options: {list(mapping.keys())}"
            )

    _check(data_key,    DATA_SOURCES,   "data_source")
    _check(period_key,  PERIODS,        "period")
    _check(horizon_key, HORIZON_SETS,   "horizon_set")
    _check(wfo_key,     WFO_SETTINGS,   "wfo_setting")
    _check(reflex_key,  REFLEX_WINDOWS, "reflex_window")
    _check(alpha_key,   RIDGE_ALPHAS,   "ridge_alpha")

    data_source = DATA_SOURCES[data_key]
    period      = PERIODS[period_key]
    horizon_set = HORIZON_SETS[horizon_key]
    wfo         = WFO_SETTINGS[wfo_key]
    reflex      = REFLEX_WINDOWS[reflex_key]
    ridge       = RIDGE_ALPHAS[alpha_key]

    # Build output path
    out_dir = os.path.join(OUTPUT_ROOT, ACTIVE_EXPERIMENT)
    os.makedirs(out_dir, exist_ok=True)

    return {
        # Keys for traceability
        "experiment_id":  ACTIVE_EXPERIMENT,
        "data_key":       data_key,
        "period_key":     period_key,
        "horizon_key":    horizon_key,
        "wfo_key":        wfo_key,
        "reflex_key":     reflex_key,
        "alpha_key":      alpha_key,

        # Resolved values
        "demo":           data_source["demo"],
        "data_label":     data_source["label"],

        "start_date":     period["start"],
        "end_date":       period["end"],
        "period_label":   period["label"],

        "horizons":       horizon_set["horizons"],
        "horizon_label":  horizon_set["label"],

        "initial_train":  wfo["initial_train"],
        "step":           wfo["step"],
        "wfo_label":      wfo["label"],

        "reflex_window":  reflex["reflex_window"],
        "zscore_window":  reflex["zscore_window"],
        "reflex_label":   reflex["label"],

        "alpha":          ridge["alpha"],
        "alpha_label":    ridge["label"],

        "output_dir":     out_dir,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = parse_active()
    print("=" * 60)
    print("ACTIVE EXPERIMENT")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k:<16} {v}")
    print("=" * 60)
    print(f"\nOutput will be saved to:\n  {cfg['output_dir']}")