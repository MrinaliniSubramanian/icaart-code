#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLAN_ORDER = ["makespan", "IV", "MTD", "SumTD", "Leximin"]

PLAN_ALIASES = {
    "baseline": "makespan",
    "base": "makespan",
    "makespan": "makespan",
    "mksp": "makespan",
    "metric1": "IV",
    "iv": "IV",
    "IV": "IV",
    "metric2": "MTD",
    "mtd": "MTD",
    "MTD": "MTD",
    "metric3": "SumTD",
    "sumtd": "SumTD",
    "SumTD": "SumTD",
    "sum_td": "SumTD",
    "metric4": "Leximin",
    "leximin": "Leximin",
    "Leximin": "Leximin",
    "lex": "Leximin",
    "krobust": "krobust",
    "k-robust": "krobust",
    "k_robust": "krobust",
    "robust": "krobust",
}


COLUMN_ALIASES = {
    "instance": [
        "instance", "instance_name", "problem", "param", "param_name",
        "instance_id", "map_instance"
    ],
    "plan": [
        "plan", "plan_type", "method", "metric", "solver", "solution_type"
    ],
    "k": [
        "delay_duration", "k", "delay_k", "delay_size", "delay"
    ],
    "planned_makespan": [
        "planned_makespan", "plan_makespan", "horizon_moves",
        "makespan", "M", "planned_M", "optimal_makespan"
    ],
    "executed_makespan": [
        "executed_makespan", "exec_makespan", "actual_makespan",
        "final_makespan", "realized_makespan", "realised_makespan",
        "execution_makespan", "observed_makespan", "M_exec"
    ],
    "total_cascade": [
        "total_cascade", "cascading_delay", "cascade", "total_propagated_delay",
        "propagated_delay", "cascade_delay"
    ],
    "delayed_agent": [
        "delayed_agent", "agent", "agent_id", "delay_agent"
    ],
    "delay_step": [
        "delay_timestep", "delay_step", "timestep", "time_step", "step", "t", "delay_t"
    ],
    "execution_id": [
        "injection_id", "execution_id", "sim_id", "simulation_id",
        "run_id", "trial_id", "delay_id", "sim"
    ],
}


def find_column(df, canonical_name, required=False):
    candidates = COLUMN_ALIASES[canonical_name]
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(
            f"Could not find required column for '{canonical_name}'.\n"
            f"Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None



def standardize_columns(raw):
    col_instance = find_column(raw, "instance", required=True)
    col_plan = find_column(raw, "plan", required=True)
    col_k = find_column(raw, "k", required=True)
    col_planned = find_column(raw, "planned_makespan", required=True)
    col_cascade = find_column(raw, "total_cascade", required=True)
    col_exec = find_column(raw, "executed_makespan", required=False)
    col_agent = find_column(raw, "delayed_agent", required=False)
    col_step = find_column(raw, "delay_step", required=False)
    col_exec_id = find_column(raw, "execution_id", required=False)

    df = pd.DataFrame()
    df["instance"] = raw[col_instance].astype(str)
    df["plan"] = raw[col_plan].astype(str).map(lambda x: PLAN_ALIASES.get(x, x))
    df["k"] = pd.to_numeric(raw[col_k], errors="coerce")
    df["planned_makespan"] = pd.to_numeric(raw[col_planned], errors="coerce")
    df["total_cascade"] = pd.to_numeric(raw[col_cascade], errors="coerce")

    if col_exec is not None:
        df["executed_makespan"] = pd.to_numeric(raw[col_exec], errors="coerce")
    else:
        df["executed_makespan"] = np.nan

    if col_agent is not None:
        df["delayed_agent"] = raw[col_agent].astype(str)
    else:
        df["delayed_agent"] = "unknown_agent"

    if col_step is not None:
        df["delay_step"] = raw[col_step].astype(str)
    else:
        df["delay_step"] = "unknown_step"

    if col_exec_id is not None:
        df["execution_id"] = raw[col_exec_id].astype(str)
    else:
        df["execution_id"] = (
            df["instance"].astype(str)
            + "_k" + df["k"].astype(str)
            + "_a" + df["delayed_agent"].astype(str)
            + "_t" + df["delay_step"].astype(str)
        )

    df = df.dropna(subset=["k", "planned_makespan", "total_cascade"])
    df["k"] = df["k"].astype(int)

    if df.empty:
        raise ValueError("No valid rows remained after parsing numeric columns.")

    return df


def add_baseline_makespan(df):
    base = df[df["plan"] == "makespan"].copy()

    if base.empty:
        raise ValueError("No rows with plan == 'makespan'. Cannot compute optimal baseline M.")

    baseline = (
        base.groupby("instance", as_index=False)["planned_makespan"]
        .min()
        .rename(columns={"planned_makespan": "M"})
    )

    out = df.merge(baseline, on="instance", how="left")

    if out["M"].isna().any():
        missing = out.loc[out["M"].isna(), "instance"].unique()
        raise ValueError(f"Could not assign baseline M for instances: {missing[:10]}")

    if (out["M"] <= 0).any():
        bad = out.loc[out["M"] <= 0, "instance"].unique()
        raise ValueError(f"Found non-positive baseline makespan M for instances: {bad[:10]}")

    out["normalized_cascade"] = out["total_cascade"] / out["M"]

    if out["executed_makespan"].notna().any():
        out["relative_executed_makespan_increase"] = (
            out["executed_makespan"] - out["M"]
        ) / out["M"]

        out["relative_extra_makespan_beyond_k"] = (
            out["executed_makespan"] - out["M"] - out["k"]
        ).clip(lower=0) / out["M"]
    else:
        out["relative_executed_makespan_increase"] = np.nan
        out["relative_extra_makespan_beyond_k"] = np.nan

    return out


def validate_same_makespan_for_diversity(df):
    
    diversity = df[df["plan"].isin(["IV", "MTD", "SumTD", "Leximin"])].copy()
    if diversity.empty:
        return

    bad = diversity[np.abs(diversity["planned_makespan"] - diversity["M"]) > 1e-9]
    if not bad.empty:
        n = len(bad)
        examples = bad[["instance", "plan", "planned_makespan", "M"]].drop_duplicates().head(10)
        print("")
        print("WARNING: Some diversity rows do not have planned_makespan equal to baseline M.")
        print(f"Number of affected rows: {n}")
        print(examples.to_string(index=False))
        print("")


def instance_level_aggregation(df):
    
    agg = (
        df.groupby(["instance", "plan", "k"], as_index=False)
        .agg(
            avg_total_cascade=("total_cascade", "mean"),
            median_total_cascade=("total_cascade", "median"),
            avg_normalized_cascade=("normalized_cascade", "mean"),
            avg_executed_makespan=("executed_makespan", "mean"),
            M=("M", "first"),
            planned_makespan=("planned_makespan", "first"),
            n_executions=("total_cascade", "size"),
        )
    )

    base = (
        agg[agg["plan"] == "makespan"]
        [["instance", "k", "avg_total_cascade", "avg_normalized_cascade"]]
        .rename(
            columns={
                "avg_total_cascade": "baseline_avg_total_cascade",
                "avg_normalized_cascade": "baseline_avg_normalized_cascade",
            }
        )
    )

    agg = agg.merge(base, on=["instance", "k"], how="left")

    agg["reduction_vs_makespan"] = (
        agg["baseline_avg_total_cascade"] - agg["avg_total_cascade"]
    )

    agg["pct_reduction_vs_makespan"] = np.where(
        agg["baseline_avg_total_cascade"] > 0,
        100.0 * agg["reduction_vs_makespan"] / agg["baseline_avg_total_cascade"],
        np.nan,
    )

  
    agg["has_nonzero_baseline_cascade"] = (
        agg["baseline_avg_total_cascade"] > 0
    )

    return agg


def make_average_table(inst):
    rows = []

    for (plan, k), sub in inst.groupby(["plan", "k"]):
        rows.append({
            "plan": plan,
            "k": int(k),
            "instances": sub["instance"].nunique(),
            "avg_cascade": sub["avg_total_cascade"].mean(),
            "median_cascade": sub["avg_total_cascade"].median(),
            "avg_normalized_cascade": sub["avg_normalized_cascade"].mean(),
            "avg_reduction_vs_makespan": sub["reduction_vs_makespan"].mean(),
            "avg_percent_reduction": sub["pct_reduction_vs_makespan"].mean(),
            "median_percent_reduction": sub["pct_reduction_vs_makespan"].median(),
            "percent_reduction_instances": sub["pct_reduction_vs_makespan"].notna().sum(),
            "zero_baseline_instances": (~sub["has_nonzero_baseline_cascade"]).sum(),
        })

    out = pd.DataFrame(rows)
    out["plan_order"] = out["plan"].map(lambda p: PLAN_ORDER.index(p) if p in PLAN_ORDER else 99)
    out = out.sort_values(["k", "plan_order", "plan"]).drop(columns=["plan_order"])
    return out


def make_clean_summary_table(avg_table):
    rows = []

    for plan in PLAN_ORDER:
        sub = avg_table[avg_table["plan"] == plan]
        if sub.empty:
            continue

        row = {"Plan": plan}

        for k in sorted(avg_table["k"].unique()):
            r = sub[sub["k"] == k]
            if r.empty:
                row[f"$\\delta={k}$"] = ""
                continue

            avg = float(r["avg_cascade"].iloc[0])
            pct = r["avg_percent_reduction"].iloc[0]

            if plan == "makespan" or pd.isna(pct):
                row[f"$\\delta={k}$"] = f"{avg:.3f} (--)"
            else:
                row[f"$\\delta={k}$"] = f"{avg:.3f} ({pct:.1f}\\%)"

        rows.append(row)

    return pd.DataFrame(rows)


def make_win_counts(inst, tolerance=1e-9):
    
    rows = []

    for (instance, k), sub in inst.groupby(["instance", "k"]):
        min_val = sub["avg_total_cascade"].min()
        for _, r in sub.iterrows():
            is_best_or_tied = r["avg_total_cascade"] <= min_val + tolerance
            rows.append({
                "instance": instance,
                "k": int(k),
                "plan": r["plan"],
                "best_or_tied": int(is_best_or_tied),
                "strict_best": 0,  
            })

        winners = sub[np.abs(sub["avg_total_cascade"] - min_val) <= tolerance]
        if len(winners) == 1:
            winner_plan = winners["plan"].iloc[0]
            rows[-len(sub):] = [
                {**row, "strict_best": int(row["plan"] == winner_plan)}
                for row in rows[-len(sub):]
            ]

    counts = (
        pd.DataFrame(rows)
        .groupby(["plan", "k"], as_index=False)
        .agg(
            best_or_tied=("best_or_tied", "sum"),
            strict_best=("strict_best", "sum"),
        )
    )

    counts["plan_order"] = counts["plan"].map(lambda p: PLAN_ORDER.index(p) if p in PLAN_ORDER else 99)
    counts = counts.sort_values(["k", "plan_order", "plan"]).drop(columns=["plan_order"])
    return counts


def plot_average_cascade(avg_table, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for plan in PLAN_ORDER:
        sub = avg_table[avg_table["plan"] == plan].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["avg_cascade"], marker="o", linewidth=2, label=plan)

    ax.set_xlabel("Injected delay duration $\\delta$")
    ax.set_ylabel("Average cascading delay")
    ax.set_title("Cascading delay under ADG execution")
    ax.set_xticks(sorted(avg_table["k"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()

    fig.savefig(out_dir / "fig1_average_cascade_vs_k.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig1_average_cascade_vs_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_win_counts(win_counts, out_dir):
    plans = [p for p in PLAN_ORDER if p in set(win_counts["plan"])]
    ks = sorted(win_counts["k"].unique())

    x = np.arange(len(plans))
    width = 0.8 / len(ks)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for i, k in enumerate(ks):
        vals = []
        for plan in plans:
            row = win_counts[(win_counts["plan"] == plan) & (win_counts["k"] == k)]
            vals.append(int(row["best_or_tied"].iloc[0]) if not row.empty else 0)

        ax.bar(x + (i - (len(ks) - 1) / 2) * width, vals, width, label=f"$\\delta={k}$")

    ax.set_xlabel("Plan")
    ax.set_ylabel("Best or tied-best instances")
    ax.set_title("Consistency of each objective")
    ax.set_xticks(x)
    ax.set_xticklabels(plans)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=len(ks))
    fig.tight_layout()

    fig.savefig(out_dir / "fig3_win_tie_counts.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_win_tie_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cdf_normalized_cascade(df, out_dir, k_filter):
    plot_df = df[df["k"] == k_filter].copy()
    title = f"CDF of normalized cascading delay, $\\delta={k_filter}$"
    filename = f"fig4_cdf_normalized_cascade_k{k_filter}"

    fig, ax = plt.subplots(figsize=(6.4, 4.1))

    global_max = plot_df["normalized_cascade"].max()
    if pd.isna(global_max):
        print(f"Skipping {filename}: no data.")
        return

    for plan in PLAN_ORDER:
        sub = plot_df[plot_df["plan"] == plan]
        if sub.empty:
            continue

        values = np.sort(sub["normalized_cascade"].to_numpy())

        x = np.r_[values[0], values]
        y = np.r_[0.0, np.arange(1, len(values) + 1) / len(values)]

        
        if values[-1] < global_max:
            x = np.r_[x, global_max]
            y = np.r_[y, 1.0]

        ax.step(x, y, where="post", label=plan)

    ax.set_xlabel("Cascading delay normalized by optimal makespan")
    ax.set_ylabel("Fraction of executions")
    ax.set_ylim(0.5, 1.01)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(clean_table, out_dir):
    latex = clean_table.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * max(0, len(clean_table.columns) - 1),
        caption=(
            "Mean cascading delay, averaged first over delay-injection "
            "executions within each instance and then across instances. "
            "Values in parentheses give the mean percentage reduction relative "
            "to the makespan-optimal plan for instances with nonzero baseline "
            "cascading delay. "
            "Lower values are better."
        ),
        label="tab:avg-cascade-summary",
    )
    (out_dir / "table_summary_clean.tex").write_text(latex)


def main():
    parser = argparse.ArgumentParser(description="Generate cascading-delay plots and Table 3.")
    parser.add_argument("--input", required=True, help="Path to adg_delay_details.csv.")
    parser.add_argument("--out-dir", default="precise_adg_plots")
    parser.add_argument("--make-cdf-per-k", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--best-plan", help=argparse.SUPPRESS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = standardize_columns(pd.read_csv(args.input))
    if "krobust" in set(df["plan"]):
        df = df[df["plan"].isin(PLAN_ORDER)].copy()
    df = add_baseline_makespan(df)
    validate_same_makespan_for_diversity(df)
    if not set(PLAN_ORDER[1:]).intersection(df["plan"]):
        raise ValueError("No diversity plans found in the input.")

    inst = instance_level_aggregation(df)
    avg_table = make_average_table(inst)
    clean_table = make_clean_summary_table(avg_table)
    clean_table.to_csv(out_dir / "table_summary_clean.csv", index=False)
    write_latex_table(clean_table, out_dir)

    plot_average_cascade(avg_table, out_dir)
    plot_win_counts(make_win_counts(inst), out_dir)
    for k in sorted(df["k"].unique()):
        plot_cdf_normalized_cascade(df, out_dir, k_filter=int(k))

    print(f"Saved cascading-delay plots and Table 3 in: {out_dir}")


if __name__ == "__main__":
    main()
