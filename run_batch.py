"""
Batch robustness runs (Step 5). Run ONLY after steps 1-4 are verified.
Grid: periods x reflex windows x ridge alphas, with horizons/WFO held fixed
at the baseline (standard horizons, 'fast' WFO). No picking the best cell --
the output is the full grid plus a stability summary.

    python run_batch.py
"""
import os
import traceback
import pandas as pd

import config
import retail_flow_engine as E

DATA      = "real"
HORIZONS  = "standard"
WFO       = "fast"
PERIODS   = ["full", "post2020", "covid", "recent"]
WINDOWS   = ["w40", "w60", "w90"]
ALPHAS    = ["low", "standard", "high"]      # 0.1 / 1.0 / 5.0


def main():
    all_incr, all_summary, failures = [], [], []
    grid = [(p, w, a) for p in PERIODS for w in WINDOWS for a in ALPHAS]
    for k, (p, w, a) in enumerate(grid, 1):
        exp_id = f"{DATA}__{p}__{HORIZONS}__{WFO}__{w}__{a}"
        print(f"\n######## [{k}/{len(grid)}] {exp_id}")
        config.ACTIVE_EXPERIMENT = exp_id
        try:
            cfg = config.parse_active()
            summary_df, incr_df, _ = E.run(cfg)
        except Exception as exc:                 # infeasible WFO combo, etc. -- log, don't hide
            traceback.print_exc()
            failures.append({"experiment_id": exp_id, "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        meta = {"experiment_id": exp_id, "period": p, "reflex_window": cfg["reflex_window"],
                "alpha": cfg["alpha"]}
        all_incr.append(incr_df.assign(**meta))
        all_summary.append(summary_df.assign(**meta))

    out_dir = os.path.join(config.OUTPUT_ROOT, "_batch_robustness")
    os.makedirs(out_dir, exist_ok=True)
    incr = pd.concat(all_incr, ignore_index=True) if all_incr else pd.DataFrame()
    summ = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()

    # Stability summary: how often does each challenger beat har_rv_vix, by horizon?
    stab = pd.DataFrame()
    if not incr.empty:
        incr["mse_lower"] = incr["mean_sq_err_diff"] < 0
        stab = (incr.groupby(["comparison", "horizon"])
                    .agg(n_runs=("experiment_id", "nunique"),
                         share_mse_lower=("mse_lower", "mean"),
                         share_sig_5pct_better=("challenger_better", "mean"),
                         median_oos_r2_vs_bench=("oos_r2_vs_bench", "median"),
                         median_p_value=("p_value", "median"))
                    .reset_index())
    path = os.path.join(out_dir, "batch_robustness.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        stab.round(4).to_excel(w, sheet_name="Stability", index=False)
        incr.round(4).to_excel(w, sheet_name="All_Incremental", index=False)
        summ.round(4).to_excel(w, sheet_name="All_WFO_Summary", index=False)
        pd.DataFrame(failures).to_excel(w, sheet_name="Failed_Runs", index=False)
    print(f"\n✓ Batch done: {len(all_incr) and len(grid) - len(failures)}/{len(grid)} runs OK -> {path}")
    if failures:
        print("Failed/infeasible runs:", *[f["experiment_id"] for f in failures], sep="\n  ")
    print(stab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
