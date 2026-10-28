"""
================================================================================
RETAIL FLOW AS A HIDDEN RISK FACTOR
================================================================================

Implements Strategy 3 from the chat with Paris: instead of using retail order
flow as a stock-picking signal, test whether the GAP between retail's
historical buy-the-dip response and its actual, realized response has
predictive value for FUTURE volatility / drawdown over several horizons.

USAGE
-----
    # Set ACTIVE_EXPERIMENT in config.py, then:
    python retail_flow_engine.py

Or import and call directly:
    from retail_flow_engine import run
    from config import parse_active
    run(parse_active())

DATA CONSTRAINT (put this in the paper's data section)
--------------------------------------------------------
No WRDS/TAQ access ⇒ no trade-level retail classification (e.g. Boehmer–
Jones–Zhang). Proxy used instead, built entirely from free sources:
  - Retail-flow proxy: FINRA's daily Reg SHO short-volume files, aggregated
    market-wide. The FINRA ADF/OTC tape share of total volume proxies
    wholesaler-internalized (overwhelmingly retail) flow — same idea as
    SEC's MIDAS off-exchange metrics.
  - Market benchmark: SPY price/returns + VIX, via yfinance.
This is a MARKET-WIDE index, not single-name — consistent with treating it
as a portfolio-level risk factor rather than a stock-picking signal.

LOOKAHEAD PROTECTION
---------------------
• Feature row at date t uses only data dated <= t (rolling windows ending at
  t, or values dated <= t — never a centered or trailing-into-the-future window).
• target_rv_h at row t = realized volatility computed from returns
  t+1 ... t+h only — strictly future relative to the feature row. Verified
  independently by assert_no_lookahead() before every run.
• The reflex beta (historical buy-the-dip slope) at t is estimated only from
  (return_s, retail-response_{s+1}) pairs fully observed BEFORE day t — the
  pair for day t itself is never inside its own training window.
• Walk-forward refits train on returns.iloc[:start] only (strictly before the
  test block about to be predicted) — expanding window, no shuffling, no
  k-fold leakage across time.
================================================================================
"""

from __future__ import annotations

import io
import os
import time
import datetime as dt
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# NOTE: intentionally NOT blanket-suppressing RuntimeWarning — that was
# hiding "Mean of empty slice" / "invalid value encountered" warnings that
# would otherwise have flagged the empty-array -> NaN bug at its source.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle,
        Paragraph, Spacer,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

TRADING_DAYS = 252

# Module-level cache dir for FINRA files (independent of any single cfg's
# output_dir, so re-running different experiments doesn't re-download data).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_MODULE_DIR, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

FINRA_URL_TEMPLATE = "https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt".replace(
    "YYYYMMDD", "{date}"
)
# FINRA also publishes an ADF/ORF-only file for some periods under a slightly
# different name; if the combined file 404s we fall back to trying that name.
FINRA_URL_TEMPLATE_ALT = "https://cdn.finra.org/equity/regsho/daily/FNRAshvolYYYYMMDD.txt".replace(
    "YYYYMMDD", "{date}"
)
# --------------------------------------------------------------------------
# NOTE on OFF_EXCHANGE_MARKET_CODE (kept for reference / backward-compat):
# The ADF ("D") was a reasonable off-exchange proxy years ago, but the ADF
# has carried essentially ZERO reported volume for the entire 2018-2024
# study window (FINRA's own daily ADF file is a ~65-byte, data-free stub on
# every date we checked). Filtering on Market == "D" therefore returns 0
# every single day -> off_exchange_share is a hard constant 0.0 -> its
# rolling z-score divides 0/0 -> NaN on every row -> dropna() wipes the
# whole table -> "0 usable rows" WFO error, regardless of period/WFO knobs.
# See fetch_finra_retail_proxy() below for the fix (use ALL reporting
# facilities in the file, not just the now-dead ADF).
# --------------------------------------------------------------------------
OFF_EXCHANGE_MARKET_CODE = "D"  # FINRA ADF / OTC tape (legacy; effectively dead post-~2014)

# cdn.finra.org sits behind Akamai, which routinely 403s requests that don't
# look like a browser (including the default python-requests user-agent).
# Without this header EVERY request 403s and gets treated identically to
# "no file for this date" -> the whole run silently produces zero rows with
# no diagnostic. Sending a real User-Agent + retries fixes that.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
if REQUESTS_AVAILABLE:
    _session = requests.Session()
    _session.headers.update(_HEADERS)
else:
    _session = None


# ============================================================================
# DATA LAYER
# ============================================================================

def _fetch_finra_file(date: dt.date, timeout: int = 20, retries: int = 2):
    """Download+parse one day's FINRA Reg SHO short-volume file, with local
    caching.

    Returns (df_or_None, status) where status is one of:
        "cache_hit", "ok", "no_data" (weekend/holiday/empty),
        "http_error", "network_error", "parse_error"
    A None df + "no_data" is expected/benign for weekends & holidays and must
    NOT crash the pipeline. http_error / network_error / parse_error are NOT
    benign and are surfaced to the caller so a broken run is obvious instead
    of silently looking like a stretch of holidays.
    """
    if not REQUESTS_AVAILABLE:
        return None, "network_error: requests not installed"
    ymd = date.strftime("%Y%m%d")
    cache_path = os.path.join(CACHE_DIR, f"finra_{ymd}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path), "cache_hit"

    last_status = None
    for template in (FINRA_URL_TEMPLATE, FINRA_URL_TEMPLATE_ALT):
        url = template.format(date=ymd)
        for attempt in range(retries + 1):
            try:
                resp = _session.get(url, timeout=timeout)
            except requests.RequestException as exc:
                last_status = f"network_error: {exc}"
                time.sleep(0.5 * (attempt + 1))
                continue
            if resp.status_code == 404:
                last_status = "no_data"  # genuinely no file for this date (weekend/holiday)
                break  # try next template rather than retrying a 404
            if resp.status_code != 200:
                last_status = f"http_error: {resp.status_code} {resp.text[:120]!r}"
                time.sleep(0.5 * (attempt + 1))
                continue
            if len(resp.content) < 50:
                last_status = "no_data"
                break
            try:
                df = pd.read_csv(io.StringIO(resp.text), sep="|")
            except Exception as exc:
                last_status = f"parse_error: {exc}"
                break
            if "Symbol" not in df.columns or "TotalVolume" not in df.columns:
                last_status = f"parse_error: unexpected columns {list(df.columns)}"
                break
            df = df[df["Symbol"].astype(str).str.len() <= 8]
            for col in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df.to_parquet(cache_path)
            return df, "ok"
    return None, (last_status or "no_data")


def fetch_finra_retail_proxy(start: str, end: str, sleep_between_calls: float = 1.2,
                              verbose: bool = True, return_log: bool = False):
    """Build the market-wide daily retail-flow proxy series.

    Returns a DataFrame indexed by date with columns:
        off_exchange_share : avg. off-exchange (ADF+TRF+ORF) TotalVolume per
                              reporting symbol that day, i.e. a market-wide
                              retail/wholesaler-internalization INTENSITY
                              level (not a 0-1 share -- see note below).
        short_volume_ratio : aggregate ShortVolume / TotalVolume, that day
        n_symbols          : number of symbols reported that day (sanity check)

    NOTE ON off_exchange_share
    ---------------------------
    Every row in a FINRA Reg SHO daily short-volume file is *already*
    off-exchange (it only covers trades reported to a TRF/ADF/ORF -- lit
    exchanges never appear in this file at all). So there is no "on-exchange
    total" inside this file to divide by, and the ADF-only slice (the old
    `Market == "D"` filter) is structurally zero because the ADF itself has
    carried ~no volume since well before 2018. Getting a true "off-exchange
    share of ALL market volume" would require a second, independent
    consolidated-tape volume source (e.g. SEC MIDAS) that this pipeline does
    not currently fetch.
    Until that's added, we instead use total off-exchange volume normalized
    by the number of symbols reporting that day (so growth in the number of
    listed/reporting symbols over 2018-2024 doesn't masquerade as a trend).
    This level is not itself bounded in [0, 1], but that's fine: downstream
    code only ever consumes its rolling z-score (`retail_share_z`) and
    day-over-day change (`retail_share_delta`), both of which only care that
    the series is a genuine, time-varying measure of off-exchange activity
    -- which this now is, whereas the old ADF-only version was a hard 0.0
    every day.
    """
    dates = pd.bdate_range(start, end)
    rows = []
    status_log: list = []
    problem_counts: dict = {}
    sample_problems: list = []

    for i, d in enumerate(dates):
        day_df, status = _fetch_finra_file(d.date())
        status_log.append({"date": d.date().isoformat(), "status": status})
        if status not in ("ok", "cache_hit", "no_data"):
            key = status.split(":")[0]
            problem_counts[key] = problem_counts.get(key, 0) + 1
            if len(sample_problems) < 5:
                sample_problems.append((d.date().isoformat(), status))
        if day_df is None or day_df.empty:
            continue
        total_all = day_df["TotalVolume"].sum()
        short_all = day_df["ShortVolume"].sum()
        n_symbols = day_df["Symbol"].nunique()
        if total_all <= 0 or n_symbols <= 0:
            continue
        # total_all here already IS total off-exchange (ADF+TRF+ORF) volume
        # for the day -- this file contains nothing else. Normalize by
        # n_symbols so the level is comparable across a period where the
        # number of reporting symbols changes; see docstring note above.
        rows.append({
            "date": d,
            "off_exchange_share": total_all / n_symbols,
            "short_volume_ratio": short_all / total_all,
            "n_symbols": n_symbols,
        })
        if verbose and (i + 1) % 100 == 0:
            print(f"  ... FINRA fetch progress: {i + 1}/{len(dates)} trading days checked, "
                  f"{len(rows)} usable so far")
        if status == "ok":          # only throttle real downloads, not cache hits
            time.sleep(sleep_between_calls)

    if problem_counts and verbose:
        print(f"  FINRA fetch had non-benign issues on some days: {problem_counts}")
        print(f"  Sample: {sample_problems}")
        if any(k == "http_error" for k in problem_counts):
            print("  -> Repeated http_error usually means FINRA/Akamai is blocking or "
                  "rate-limiting this client. Try again later, or reduce request rate.")

    if not rows:
        detail = f" First observed issues: {sample_problems}" if sample_problems else ""
        raise RuntimeError(
            f"No FINRA data could be retrieved for the requested range.{detail} "
            "Check network access to cdn.finra.org, or use a 'demo' data_source in config.py."
        )
    out = pd.DataFrame(rows).set_index("date")
    return (out, pd.DataFrame(status_log)) if return_log else out


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """Robustly pull a 'Close' Series out of a yfinance download result.

    Newer yfinance versions sometimes return MultiIndex columns (e.g.
    ('Close', 'SPY')) even for a single ticker, depending on version/call
    signature. Handle both shapes instead of assuming one — assuming the
    flat-columns shape was silently returning garbage/NaN on newer yfinance.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            sub = raw["Close"]
        else:
            sub = raw.xs("Close", axis=1, level=-1)
        if isinstance(sub, pd.DataFrame):
            sub = sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]
        return sub.squeeze().rename("close")
    return raw["Close"].squeeze().rename("close")


def fetch_market_and_vix(start: str, end: str) -> pd.DataFrame:
    """SPY close-to-close returns + VIX level, from yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for live data. Install with: "
            "pip install yfinance --break-system-packages"
        ) from exc

    # yf.download's `end` is EXCLUSIVE, so the literal last requested day was
    # silently being dropped. Push it out by one day to include it.
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        spy = yf.download("SPY", start=start, end=end_inclusive, progress=False, auto_adjust=True)
        vix = yf.download("^VIX", start=start, end=end_inclusive, progress=False, auto_adjust=True)
    except Exception as exc:
        raise RuntimeError(
            f"yfinance download failed ({exc}). Yahoo occasionally changes its API in ways "
            "that break older yfinance versions — try `pip install -U yfinance "
            "--break-system-packages` first, or use a 'demo' data_source in config.py."
        ) from exc

    if spy.empty or vix.empty:
        raise RuntimeError(
            "yfinance returned no data — check network access, the date range, or try "
            "`pip install -U yfinance --break-system-packages` (Yahoo's API shifts periodically)."
        )

    spy_close = _extract_close(spy, "SPY")
    vix_close = _extract_close(vix, "^VIX")

    df = pd.DataFrame({
        "spy_close": spy_close,
        "ret": np.log(spy_close).diff(),
        "vix": vix_close,
    }).dropna(subset=["spy_close"])
    return df


def generate_synthetic_demo_data(start: str = "2018-01-01", end: str = "2024-12-31", seed: int = 7):
    """Generate a synthetic market + retail-flow dataset with a DELIBERATE,
    partially-decaying buy-the-dip relationship baked in, so the pipeline
    (features, WFO, benchmarks) can be sanity-checked end to end without
    needing internet access to FINRA/Yahoo.

    For CODE VALIDATION ONLY — not evidence for the paper.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    vol = np.zeros(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = 0.03 * 0.01 + 0.90 * vol[t - 1] + 0.09 * abs(rng.normal(0, 0.01))
    ret = rng.normal(0, 1, n) * vol - 0.5 * vol ** 2

    spy_close = 300 * np.exp(np.cumsum(ret))
    vix = 100 * (vol / vol.mean()) * 12 + rng.normal(0, 1, n)
    vix = np.clip(vix, 9, 85)

    base_share = 0.40 + 0.03 * np.sin(np.linspace(0, 6, n))
    retail_noise = rng.normal(0, 0.01, n)

    reflex_strength = np.linspace(0.9, 0.35, n)  # ground truth, unobservable in real life
    dip = np.minimum(ret, 0.0)
    retail_response = -reflex_strength * dip * 3.0
    off_exchange_share = base_share + retail_response + retail_noise
    off_exchange_share = np.clip(off_exchange_share, 0.20, 0.65)

    short_volume_ratio = 0.45 + 0.10 * (vol / vol.mean() - 1) + rng.normal(0, 0.02, n)
    short_volume_ratio = np.clip(short_volume_ratio, 0.1, 0.8)

    market_df = pd.DataFrame({"spy_close": spy_close, "ret": ret, "vix": vix}, index=dates)
    retail_df = pd.DataFrame({
        "off_exchange_share": off_exchange_share,
        "short_volume_ratio": short_volume_ratio,
        "n_symbols": 4500,
    }, index=dates)
    return market_df, retail_df


# ============================================================================
# MASTER DAILY DATASET  (download once, reuse for every experiment)
# ============================================================================

MASTER_START, MASTER_END = "2018-01-01", "2024-12-31"   # must cover every period in config.PERIODS
MASTER_PATH    = os.path.join(CACHE_DIR, "master_daily.parquet")
QUALITY_JSON   = os.path.join(CACHE_DIR, "master_quality_report.json")
STATUS_LOG_CSV = os.path.join(CACHE_DIR, "finra_status_log.csv")
_BENIGN_STATUS = ("ok", "cache_hit", "no_data")


def build_quality_report(master: pd.DataFrame, status_log: pd.DataFrame,
                         start: str, end: str) -> dict:
    """Data-quality summary for the master dataset: trading days, FINRA
    observations, missing days, and HTTP/network/parse errors."""
    import json
    bdays = pd.bdate_range(start, end)
    mkt_days = master["spy_close"].dropna().index
    finra_days = master["off_exchange_share"].dropna().index
    both = mkt_days.intersection(finra_days)
    missing = mkt_days.difference(finra_days)

    kind = status_log["status"].str.split(":").str[0]
    # FINRA's CDN answers 403 (not 404) for dates with no file, i.e. market
    # holidays. Only fetch problems on days SPY actually traded are real errors.
    is_problem = ~kind.isin(_BENIGN_STATUS)
    on_trading_day = pd.to_datetime(status_log["date"]).isin(mkt_days)
    errors = status_log[is_problem & on_trading_day]
    miss_str = {d.strftime("%Y-%m-%d") for d in missing}
    miss_kind = kind[status_log["date"].isin(miss_str)]
    report = {
        "range": [start, end],
        "business_days_in_range": int(len(bdays)),
        "spy_trading_days": int(len(mkt_days)),
        "business_days_without_spy_(holidays)": int(len(bdays.difference(mkt_days))),
        "finra_observations": int(len(finra_days)),
        "trading_days_with_both_spy_and_finra": int(len(both)),
        "trading_days_missing_finra": int(len(missing)),
        "finra_coverage_pct": round(100 * len(both) / max(len(mkt_days), 1), 2),
        "missing_finra_dates": [d.strftime("%Y-%m-%d") for d in missing],
        "fetch_status_counts": {k: int(v) for k, v in kind.value_counts().items()},
        "n_http_network_parse_errors": int(len(errors)),
        "n_http_errors_on_non_trading_days_(benign, holidays)": int((is_problem & ~on_trading_day).sum()),
        "missing_finra_days_by_fetch_status": {k: int(v) for k, v in miss_kind.value_counts().items()},
        "error_samples": errors.head(10).to_dict("records"),
        "n_symbols_min_median_max": [
            int(master["n_symbols"].min()), int(master["n_symbols"].median()),
            int(master["n_symbols"].max())] if master["n_symbols"].notna().any() else None,
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(QUALITY_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    status_log.to_csv(STATUS_LOG_CSV, index=False)
    return report


def build_master_dataset(start: str = MASTER_START, end: str = MASTER_END,
                         force_refresh: bool = False,
                         allow_partial: bool = False) -> pd.DataFrame:
    """Return the locked local master daily dataset (SPY ret/close, VIX, FINRA
    proxy). Built ONCE from the network and saved to MASTER_PATH; every later
    run loads it from disk. Delete the file or pass force_refresh=True to rebuild.

    If any day failed with a non-benign error (HTTP / network / parse), the
    master is NOT written (so a transient failure is never locked in).
    Re-run: successfully downloaded days are cached and won't be re-fetched."""
    import json
    if os.path.exists(MASTER_PATH) and not force_refresh:
        master = pd.read_parquet(MASTER_PATH)
        if os.path.exists(QUALITY_JSON):
            with open(QUALITY_JSON, encoding="utf-8") as f:
                q = json.load(f)
            print(f"  [master] loaded {MASTER_PATH}: {q['spy_trading_days']} SPY days, "
                  f"{q['finra_observations']} FINRA obs, "
                  f"{q['trading_days_missing_finra']} missing, "
                  f"{q['n_http_network_parse_errors']} fetch errors")
        return master

    print(f"[master] building {start} -> {end} (one-off download) ...")
    market_df = fetch_market_and_vix(start, end)
    market_df.index = pd.to_datetime(market_df.index).tz_localize(None).normalize()
    retail_df, status_log = fetch_finra_retail_proxy(start, end, return_log=True)
    retail_df.index = pd.to_datetime(retail_df.index).tz_localize(None).normalize()

    master = market_df.join(retail_df, how="outer").sort_index()
    report = build_quality_report(master, status_log, start, end)

    print("\n=== DATA QUALITY REPORT ===")
    for k, v in report.items():
        if k not in ("missing_finra_dates", "error_samples"):
            print(f"  {k:<42} {v}")
    print(f"  (full report: {QUALITY_JSON}; per-day log: {STATUS_LOG_CSV})")

    if report["n_http_network_parse_errors"] > 0 and not allow_partial:
        raise RuntimeError(
            f"{report['n_http_network_parse_errors']} days failed with HTTP/network/parse errors; "
            "master NOT saved. Re-run to retry just those days (the rest is cached), "
            "or call build_master_dataset(allow_partial=True) to lock it anyway.")
    master.to_parquet(MASTER_PATH)
    print(f"  ✓  Master dataset locked: {MASTER_PATH}")
    return master


def load_data(cfg: dict):
    """Demo -> synthetic. Real -> slice the locked local master (no download)."""
    if cfg["demo"]:
        print("[demo mode] generating synthetic data — for pipeline validation only.")
        return generate_synthetic_demo_data(cfg["start_date"], cfg["end_date"])
    master = build_master_dataset()
    sl = master.loc[cfg["start_date"]:cfg["end_date"]]
    market_df = sl[["spy_close", "ret", "vix"]].dropna(subset=["spy_close"])
    retail_df = sl[["off_exchange_share", "short_volume_ratio", "n_symbols"]] \
        .dropna(subset=["off_exchange_share"])
    return market_df, retail_df


# ============================================================================
# FEATURE ENGINEERING  (no lookahead)
# ============================================================================
#
# TIMING CONVENTION (read this before touching anything)
# --------------------------------------------------------
# - Everything with a `_t` feature name uses ONLY information available as of
#   the close of day t (rolling windows ending at t, or values dated <= t).
# - The target `target_rv_h` at row t is realized vol from returns
#   t+1 ... t+h — i.e. STRICTLY FUTURE relative to the feature row.
# - The last h rows of every horizon's table are dropped (their target would
#   need data past the end of the sample) rather than filled or bfilled.
# - The "reflex" features are built only from (return_s, next-day retail
#   response_{s+1}) pairs where s+1 <= t, i.e. pairs already fully observed
#   by day t. The rolling regression estimating the buy-the-dip beta at t
#   never uses pair (t, t+1).

def _rolling_ols_slope(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope of y on x (single regressor + intercept), window ending at each t."""
    x = x.values
    y = y.values
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        xs = x[i - window: i]
        ys = y[i - window: i]
        mask = ~(np.isnan(xs) | np.isnan(ys))
        if mask.sum() < window // 2:
            continue
        xs, ys = xs[mask], ys[mask]
        xm, ym = xs.mean(), ys.mean()
        denom = ((xs - xm) ** 2).sum()
        if denom <= 1e-12:
            continue
        beta = ((xs - xm) * (ys - ym)).sum() / denom
        out[i - 1] = beta
    return pd.Series(out, index=range(n))


def build_feature_target_table(
    market_df: pd.DataFrame,
    retail_df: pd.DataFrame,
    horizons=(5, 10, 20),
    reflex_window: int = 60,
    zscore_window: int = 60,
) -> dict:
    m_df = market_df.copy()
    r_df = retail_df.copy()
    m_df.index = pd.to_datetime(m_df.index).tz_localize(None).normalize()
    r_df.index = pd.to_datetime(r_df.index).tz_localize(None).normalize()

    df = m_df.join(r_df, how="inner").sort_index()

    # ── Alignment & Missing Data Fill ──
    df["off_exchange_share"] = df["off_exchange_share"].ffill(limit=3)
    df["short_volume_ratio"] = df["short_volume_ratio"].ffill(limit=3)
    df = df.dropna(subset=["ret"]).copy()

    print(f"  [diag] Clean aligned rows available for WFO: {len(df)}")
    if len(df) == 0:
        raise RuntimeError(
            "Index alignment resulted in 0 rows. Check date ranges in market_df vs. retail_df."
        )

    # ---- base (benchmark / HAR-style) features -------------------------------
    for w in (5, 10, 20):
        df[f"rv_{w}"] = df["ret"].rolling(w).std(ddof=0) * np.sqrt(TRADING_DAYS)
    df["vix_level"] = df["vix"]
    df["vix_change_5"] = df["vix"].diff(5)

    # ---- retail-flow proxy features ------------------------------------------
    df["retail_share"] = df["off_exchange_share"]
    roll_mean = df["retail_share"].rolling(zscore_window).mean()
    roll_std = df["retail_share"].rolling(zscore_window).std(ddof=0)
    df["retail_share_z"] = (df["retail_share"] - roll_mean) / roll_std.replace(0, np.nan)
    df["retail_short_ratio"] = df["short_volume_ratio"]

    # next-day change in retail share
    df["retail_share_delta"] = df["retail_share"].diff()

    resp = df["retail_share_delta"]
    driver = df["ret"].shift(1)
    reflex_beta_raw = _rolling_ols_slope(
        driver.shift(1).reset_index(drop=True),
        resp.shift(1).reset_index(drop=True),
        reflex_window,
    )
    df["reflex_beta"] = reflex_beta_raw.values

    df["predicted_response"] = df["reflex_beta"] * driver
    df["reflex_gap"] = resp - df["predicted_response"]
    df["reflex_gap_roll_mean"] = df["reflex_gap"].rolling(20).mean()

    # ── Εστιασμένο σήμα πίεσης retail (μόνο σε πτωτικές ημέρες) ──
    df["reflex_gap_stress"] = np.minimum(df["reflex_gap"], 0.0) * (df["ret"] < 0).astype(float)

    feature_cols_base = ["rv_5", "rv_10", "rv_20", "vix_level", "vix_change_5"]
    feature_cols_retail = [
        "retail_share_z",
        "reflex_gap_stress"
    ]

    feature_cols_extra = ["retail_short_ratio"]   # used ONLY by ablation C

    out = {}
    for h in horizons:
        fwd_ret = df["ret"].shift(-1).rolling(h).apply(lambda a: np.std(a, ddof=0), raw=True)
        target = fwd_ret.shift(-(h - 1)) * np.sqrt(TRADING_DAYS)

        table = df[feature_cols_base + feature_cols_retail + feature_cols_extra].copy()
        table["target_rv_h"] = target
        table = table.dropna()
        out[h] = table

    meta = {
        "feature_cols_base": feature_cols_base,
        "feature_cols_retail": feature_cols_retail,
        "feature_cols_extra": feature_cols_extra,
        "raw": df,
    }
    return out, meta

def usable_rows_table(tables: Dict[int, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    rows = []
    for h, t in tables.items():
        n = len(t)
        n_oos = max(0, n - cfg["initial_train"])
        rows.append({
            "horizon": h, "usable_rows": n, "initial_train": cfg["initial_train"],
            "oos_rows": n_oos, "refit_blocks": int(np.ceil(n_oos / cfg["step"])),
            "first_date": t.index.min().date(), "last_date": t.index.max().date(),
        })
    return pd.DataFrame(rows)


def assert_wfo_feasible(tables: Dict[int, pd.DataFrame], cfg: dict, min_oos_blocks: int = 3):
    """Raise a clear error BEFORE running if a (period, wfo) combo leaves too
    few — or zero — out-of-sample rows, instead of letting walk_forward_run
    silently return empty arrays that turn into all-NaN metrics.

    This is the check the config.py comments describe but that was never
    actually implemented — without it, walk_forward_run's `while start < n`
    loop simply never executes when initial_train_size >= n, preds stays
    all-NaN, the NaN-mask leaves empty y_true/y_pred arrays, and
    np.mean(empty array) -> NaN gets swallowed by the module-level
    warnings.filterwarnings("ignore") instead of surfacing a warning.
    """
    initial_train = cfg["initial_train"]
    step = cfg["step"]
    problems = []
    for h, table in tables.items():
        n = len(table)
        n_oos = max(0, n - initial_train)
        n_blocks = n_oos // step if step > 0 else 0
        if n <= initial_train:
            problems.append(
                f"  horizon={h}: only {n} usable rows after warm-up/target "
                f"drop, but initial_train={initial_train} -> ZERO out-of-sample "
                f"rows (walk_forward_run's loop never executes)."
            )
        elif n_blocks < min_oos_blocks:
            problems.append(
                f"  horizon={h}: {n} usable rows, initial_train={initial_train}, "
                f"step={step} -> only {n_blocks} walk-forward refit block(s) "
                f"({n_oos} OOS rows total). Metrics will be present but noisy/"
                f"near-meaningless with so few blocks (want >= {min_oos_blocks})."
            )
    if any("ZERO" in p for p in problems):
        raise ValueError(
            "WFO configuration infeasible for experiment "
            f"'{cfg['experiment_id']}':\n" + "\n".join(problems) +
            "\n\nFix: pick a shorter initial_train (e.g. wfo_key='fast') or a "
            "longer period (e.g. period_key='full'/'post2020') so that "
            "initial_train < usable rows for every horizon. See the row-count "
            "guidance in config.py's WFO_SETTINGS comment."
        )
    if problems:
        print("  ⚠  WFO feasibility warning for experiment "
              f"'{cfg['experiment_id']}':")
        print("\n".join(problems))
        print("  Continuing, but treat these horizons' metrics with caution.")


def assert_no_lookahead(market_df: pd.DataFrame, retail_df: pd.DataFrame, cfg: dict):
    """Re-derive the target construction independently and confirm alignment.
    Raises AssertionError on failure."""
    tables, meta = build_feature_target_table(
        market_df, retail_df, cfg["horizons"],
        reflex_window=cfg["reflex_window"], zscore_window=cfg["zscore_window"],
    )
    df = meta["raw"]
    ret = df["ret"]

    for h, table in tables.items():
        common_idx = table.index
        sample_idx = common_idx[len(common_idx) // 2: len(common_idx) // 2 + 5]
        for t in sample_idx:
            pos = df.index.get_loc(t)
            if pos + h >= len(df):
                continue
            manual_future = ret.iloc[pos + 1: pos + 1 + h].values
            manual_target = np.std(manual_future, ddof=0) * np.sqrt(TRADING_DAYS)
            got = table.loc[t, "target_rv_h"]
            assert np.isclose(manual_target, got, rtol=1e-6), (
                f"Look-ahead check FAILED at horizon {h}, date {t}: "
                f"expected {manual_target}, got {got}"
            )
    print("  ✓  No-look-ahead sanity check passed for horizons:", cfg["horizons"])


# ============================================================================
# WALK-FORWARD BACKTEST ENGINE  (no lookahead)
# ============================================================================
#
# - Strictly chronological expanding-window WFO: the model is refit every
#   `step` trading days, and each refit is trained ONLY on data strictly
#   before the test block it's about to predict. No shuffling, no k-fold.
# - Nested model specs per horizon:
#     naive_persistence  : predict target = rv_20 (today's trailing vol)
#     har_rv             : rv_5, rv_10, rv_20                (classic HAR-RV)
#     har_rv_vix         : + vix_level, vix_change_5          (implied vol)
#     har_rv_vix_retail  : + all retail-flow / reflex features (the claim)

SPEC_ORDER = [
    "naive_persistence", "har_rv", "har_rv_vix",
    "ablation_retail_share_z", "ablation_reflex_gap_stress",
    "ablation_short_volume_ratio", "har_rv_vix_retail",
]


@dataclass
class WFOResult:
    horizon: int
    spec_name: str
    dates: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray

    def metrics(self, baseline_pred: np.ndarray = None) -> dict:
        err = self.y_true - self.y_pred
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        if baseline_pred is not None:
            sse_model = np.sum(err ** 2)
            sse_baseline = np.sum((self.y_true - baseline_pred) ** 2)
            oos_r2 = 1 - sse_model / sse_baseline if sse_baseline > 0 else np.nan
        else:
            oos_r2 = np.nan
        return {"horizon": self.horizon, "spec": self.spec_name,
                "MAE": mae, "RMSE": rmse, "OOS_R2_vs_naive": oos_r2}


def _fit_predict_ridge(X_train, y_train, X_test, alpha=1.0):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = Ridge(alpha=alpha)
    model.fit(Xtr, y_train)
    return model.predict(Xte)


def walk_forward_run(
    table: pd.DataFrame, feature_cols: List[str], target_col: str, horizon: int,
    spec_name: str, initial_train_size: int = 500, step: int = 21, alpha: float = 1.0,
) -> WFOResult:
    """Expanding-window walk-forward: refit every `step` days, predict the next block."""
    table = table.sort_index()
    X_all = table[feature_cols].values
    y_all = table[target_col].values
    dates = table.index

    n = len(table)
    preds = np.full(n, np.nan)

    start = initial_train_size
    while start < n:
        end_test = min(start + step, n)
        # PURGE: target_rv_h at row r uses returns r+1..r+h, so it is only fully
        # realized at row r+h. At prediction row `start`, only rows r <= start-h
        # have known targets -> train on [:start-h+1], not [:start] (which would
        # let training targets overlap the test block's future returns).
        train_end = start - horizon + 1
        assert train_end - 1 + horizon <= start, "WFO leakage: training target overlaps test block"
        X_train, y_train = X_all[:train_end], y_all[:train_end]
        X_test = X_all[start:end_test]
        if len(X_train) < 30:
            start = end_test
            continue
        preds[start:end_test] = _fit_predict_ridge(X_train, y_train, X_test, alpha=alpha)
        start = end_test

    mask = ~np.isnan(preds)
    return WFOResult(horizon=horizon, spec_name=spec_name,
                      dates=dates[mask], y_true=y_all[mask], y_pred=preds[mask])


def naive_persistence_result(table: pd.DataFrame, target_col: str, horizon: int,
                              initial_train_size: int = 500) -> WFOResult:
    """Predicts target(t) = rv_20(t) — 'tomorrow's vol looks like today's trailing vol'.
    Needs no fitting, just the same evaluation window as the fitted models."""
    table = table.sort_index()
    y_all = table[target_col].values
    pred_all = table["rv_20"].values
    dates = table.index
    sl = slice(initial_train_size, len(table))
    return WFOResult(horizon=horizon, spec_name="naive_persistence",
                      dates=dates[sl], y_true=y_all[sl], y_pred=pred_all[sl])


def run_all_specs(
    tables: Dict[int, pd.DataFrame], feature_cols_base: List[str],
    feature_cols_retail: List[str], initial_train_size: int = 500,
    step: int = 21, alpha: float = 1.0,
) -> Dict[int, Dict[str, WFOResult]]:
    """Runs the full spec ladder (naive -> har_rv -> har_rv_vix -> +retail) for every horizon."""
    results: Dict[int, Dict[str, WFOResult]] = {}
    rv_only = [c for c in feature_cols_base if c.startswith("rv_")]
    rv_vix = feature_cols_base
    rv_vix_retail = feature_cols_base + feature_cols_retail

    for h, table in tables.items():
        results[h] = {}
        results[h]["naive_persistence"] = naive_persistence_result(
            table, "target_rv_h", h, initial_train_size)
        results[h]["har_rv"] = walk_forward_run(
            table, rv_only, "target_rv_h", h, "har_rv", initial_train_size, step, alpha)
        results[h]["har_rv_vix"] = walk_forward_run(
            table, rv_vix, "target_rv_h", h, "har_rv_vix", initial_train_size, step, alpha)
        results[h]["har_rv_vix_retail"] = walk_forward_run(
            table, rv_vix_retail, "target_rv_h", h, "har_rv_vix_retail", initial_train_size, step, alpha)

        # ── Ablations: HAR-RV-VIX + ONE retail component at a time ──
        ablations = {
            "ablation_retail_share_z":     rv_vix + ["retail_share_z"],
            "ablation_reflex_gap_stress":  rv_vix + ["reflex_gap_stress"],
            "ablation_short_volume_ratio": rv_vix + ["retail_short_ratio"],
        }
        for name, cols in ablations.items():
            results[h][name] = walk_forward_run(
                table, cols, "target_rv_h", h, name, initial_train_size, step, alpha)
    return results


# ============================================================================
# EVALUATION / BENCHMARKS
# ============================================================================

def summarize(results: Dict[int, Dict[str, WFOResult]]) -> pd.DataFrame:
    """OOS R² (relative to same-period naive_persistence baseline), MAE, RMSE
    for every spec and horizon."""
    rows = []
    for h, spec_dict in results.items():
        baseline = spec_dict["naive_persistence"]
        base_series = pd.Series(baseline.y_pred, index=baseline.dates)
        for name, res in spec_dict.items():
            baseline_aligned = base_series.reindex(res.dates).values
            rows.append(res.metrics(baseline_pred=baseline_aligned))
    return pd.DataFrame(rows).sort_values(["horizon", "spec"]).reset_index(drop=True)


def _dm_hac(d: np.ndarray, h: int):
    """Diebold-Mariano test on a loss differential d_t with a Newey-West (Bartlett)
    HAC long-run variance, truncation lag h-1 (h-step overlapping targets give
    MA(h-1) errors), plus the Harvey-Leybourne-Newbold small-sample correction
    (Student-t with n-1 df). Returns (dm_stat, dm_stat_hln, p_value_hln, lag)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    lag = max(h - 1, 0)
    dbar = d.mean()
    u = d - dbar
    lrv = float(np.dot(u, u)) / n
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(np.dot(u[k:], u[:-k])) / n
    if lrv <= 0:
        return np.nan, np.nan, np.nan, lag
    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return dm, dm_hln, p, lag


def incremental_value_test(
    results: Dict[int, Dict[str, WFOResult]],
    benchmark: str = "har_rv_vix",
    challengers=("har_rv_vix_retail", "ablation_retail_share_z",
                 "ablation_reflex_gap_stress", "ablation_short_volume_ratio"),
) -> pd.DataFrame:
    """HAC / Diebold-Mariano test on squared-error differentials,
    challenger minus benchmark. Negative mean diff + small p-value => the
    challenger reduces forecast error beyond the benchmark. The naive paired
    t-test p-value is kept as a column only to show the effect of the correction."""
    rows = []
    for h, spec_dict in results.items():
        b = spec_dict[benchmark]
        for name in challengers:
            if name not in spec_dict:
                continue
            a = spec_dict[name]
            common = a.dates.intersection(b.dates)
            ea = pd.Series(a.y_true - a.y_pred, index=a.dates).reindex(common) ** 2
            eb = pd.Series(b.y_true - b.y_pred, index=b.dates).reindex(common) ** 2
            diff = (ea - eb).dropna()
            row = {"comparison": f"{name} vs {benchmark}", "horizon": h, "n_obs": len(diff)}
            if len(diff) < 10:
                row.update({"hac_lag": max(h - 1, 0), "mean_sq_err_diff": np.nan,
                            "oos_r2_vs_bench": np.nan, "dm_stat": np.nan,
                            "dm_stat_hln": np.nan, "p_value": np.nan,
                            "p_naive_ttest": np.nan, "challenger_better": False})
                rows.append(row)
                continue
            dm, dm_hln, p, lag = _dm_hac(diff.values, h)
            _, p_naive = stats.ttest_1samp(diff.values, 0.0)
            row.update({
                "hac_lag": lag,
                "mean_sq_err_diff": diff.mean(),
                "oos_r2_vs_bench": 1.0 - ea.loc[diff.index].sum() / eb.loc[diff.index].sum(),
                "dm_stat": dm, "dm_stat_hln": dm_hln,
                "p_value": p, "p_naive_ttest": p_naive,
                "challenger_better": bool(diff.mean() < 0 and p < 0.05),
            })
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# OUTPUT  (plots, Excel, PDF)
# ============================================================================

def save_plots(meta: dict, summary_df: pd.DataFrame, cfg: dict) -> None:
    raw = meta["raw"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(raw.index, raw["spy_close"])
    axes[0].set_title("Market price (SPY or synthetic proxy)")

    axes[1].plot(raw.index, raw["retail_share"], color="darkorange")
    axes[1].set_title("Off-exchange / retail-flow share (market-wide proxy)")

    axes[2].plot(raw.index, raw["reflex_gap_roll_mean"], color="crimson")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Reflex gap (20d avg): actual vs. history-implied buy-the-dip response\n"
                       "(persistently negative & widening = reflex is lagging)")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], "retail_flow_overview.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    spec_order = [s for s in SPEC_ORDER if s in summary_df["spec"].unique()]
    pivot = summary_df.pivot(index="horizon", columns="spec", values="OOS_R2_vs_naive")[spec_order]
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("OOS R² vs. naive persistence baseline")
    ax.set_title("Walk-forward out-of-sample R² by model spec and horizon")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["output_dir"], "oos_r2_by_spec.png"), dpi=130)
    plt.close(fig)

    print(f"  ✓  Plots: {cfg['output_dir']}")


def save_excel(summary_df: pd.DataFrame, incr_df: pd.DataFrame, cfg: dict,
               rows_df: pd.DataFrame = None) -> str:
    path = os.path.join(cfg["output_dir"], "results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        summary_df.round(4).to_excel(w, sheet_name="WFO_Summary", index=False)
        incr_df.round(4).to_excel(w, sheet_name="Incremental_Value", index=False)
        if rows_df is not None:
            rows_df.to_excel(w, sheet_name="Usable_Rows", index=False)

        cfg_snap = {k: str(v) for k, v in cfg.items()}
        pd.DataFrame.from_dict(
            cfg_snap, orient="index", columns=["Value"]
        ).to_excel(w, sheet_name="Config")

    print(f"  ✓  Excel: {path}")
    return path


def save_pdf(summary_df: pd.DataFrame, incr_df: pd.DataFrame, cfg: dict) -> str | None:
    if not PDF_AVAILABLE:
        print("  ⚠  ReportLab not installed – PDF skipped.")
        return None

    path = os.path.join(cfg["output_dir"], "results.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=15, leftMargin=15,
                             topMargin=20, bottomMargin=20)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("T", parent=styles["Heading1"],
                                  fontSize=18, alignment=TA_CENTER, spaceAfter=6)
    story.append(Paragraph("Retail Flow as a Hidden Risk Factor", title_style))
    story.append(Paragraph(cfg["experiment_id"], styles["Heading3"]))
    story.append(Spacer(1, 0.15 * inch))

    # ── Config table ────────────────────────────────────────────────────
    cfg_rows = [["Parameter", "Value"]] + [
        [k.replace("_", " ").title(), str(v)]
        for k, v in {
            "Data source": cfg["data_label"],
            "Period":      cfg["period_label"],
            "Horizons":    cfg["horizon_label"],
            "Walk-fwd":    cfg["wfo_label"],
            "Reflex win":  cfg["reflex_label"],
            "Ridge alpha": cfg["alpha_label"],
        }.items()
    ]
    ct = Table(cfg_rows, colWidths=[1.8 * inch, 5.1 * inch])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2FF")]),
    ]))
    story.append(ct)
    story.append(Spacer(1, 0.2 * inch))

    # ── WFO summary table ──────────────────────────────────────────────
    story.append(Paragraph("Walk-Forward OOS Performance", styles["Heading2"]))
    cols = ["horizon", "spec", "MAE", "RMSE", "OOS_R2_vs_naive"]
    header = ["Horizon", "Spec", "MAE", "RMSE", "OOS R² vs naive"]
    rows = [[str(r[c]) if c in ("horizon", "spec") else f"{r[c]:.4f}" for c in cols]
            for _, r in summary_df.iterrows()]
    mt = Table([header] + rows, colWidths=[0.8 * inch, 1.6 * inch, 1.0 * inch, 1.0 * inch, 1.3 * inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F8E9")]),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.2 * inch))

    # ── Incremental value table ────────────────────────────────────────
    if not incr_df.empty:
        story.append(Paragraph("Incremental Value: retail features vs. har_rv_vix", styles["Heading2"]))
        icols = [c for c in incr_df.columns]
        iheader = [c.replace("_", " ").title() for c in icols]
        irows = [[f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in icols]
                 for _, r in incr_df.iterrows()]
        it = Table([iheader] + irows)
        it.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b71c1c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBE9E7")]),
        ]))
        story.append(it)

    doc.build(story)
    print(f"  ✓  PDF: {path}")
    return path


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run(cfg: dict):
    """Run one experiment defined by cfg (from config.parse_active())."""

    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {cfg['experiment_id']}")
    print(f"  Data:       {cfg['data_label']}")
    print(f"  Period:     {cfg['period_label']}")
    print(f"  Horizons:   {cfg['horizon_label']}")
    print(f"  Walk-fwd:   {cfg['wfo_label']}")
    print(f"  Reflex win: {cfg['reflex_label']}")
    print(f"  Ridge:      {cfg['alpha_label']}")
    print("=" * 70)

    # ── 1. Load data (demo or real, per cfg) ────────────────────────────
    market_df, retail_df = load_data(cfg)

    # ── 2. No-look-ahead sanity check BEFORE trusting any result ────────
    print("\nRunning no-look-ahead sanity checks...")
    assert_no_lookahead(market_df, retail_df, cfg)

    # ── 3. Build feature/target tables (per horizon) ────────────────────
    tables, meta = build_feature_target_table(
        market_df, retail_df, cfg["horizons"],
        reflex_window=cfg["reflex_window"], zscore_window=cfg["zscore_window"],
    )

    # ── 4. Walk-forward optimization across the spec ladder ─────────────
    assert_wfo_feasible(tables, cfg)
    print(f"\nRunning walk-forward optimization for horizons {cfg['horizons']} ...")
    results = run_all_specs(
        tables, meta["feature_cols_base"], meta["feature_cols_retail"],
        initial_train_size=cfg["initial_train"], step=cfg["step"], alpha=cfg["alpha"],
    )

    # ── 5. Summaries ──────────────────────────────────────────────────
    summary_df = summarize(results)
    incr_df = incremental_value_test(results)
    rows_df = usable_rows_table(tables, cfg)

    pd.set_option("display.width", 120)
    print("\n=== Walk-forward OOS performance by spec & horizon ===")
    print(summary_df.to_string(index=False))
    print("\n=== Incremental value vs. har_rv_vix (HAC / Diebold-Mariano) ===")
    print(incr_df.to_string(index=False))
    print("\n=== Usable rows per horizon ===")
    print(rows_df.to_string(index=False))

    # ── 6. Output: plots + Excel + PDF, all into cfg['output_dir'] ──────
    save_plots(meta, summary_df, cfg)
    save_excel(summary_df, incr_df, cfg, rows_df)
    save_pdf(summary_df, incr_df, cfg)

    print(f"\n  Output folder:  {cfg['output_dir']}")
    return summary_df, incr_df, results


if __name__ == "__main__":
    from config import parse_active
    cfg = parse_active()
    run(cfg)