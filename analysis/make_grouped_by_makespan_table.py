#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
    "total_cascade": [
        "total_cascade", "cascading_delay", "cascade", "total_propagated_delay",
        "propagated_delay", "cascade_delay"
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

    df = pd.DataFrame()
    df["instance"] = raw[col_instance].astype(str)
    df["plan"] = raw[col_plan].astype(str).map(lambda x: PLAN_ALIASES.get(x, x))
    df["k"] = pd.to_numeric(raw[col_k], errors="coerce")
    df["planned_makespan"] = pd.to_numeric(raw[col_planned], errors="coerce")
    df["total_cascade"] = pd.to_numeric(raw[col_cascade], errors="coerce")

    df = df.dropna(subset=["k", "planned_makespan", "total_cascade"])
    df["k"] = df["k"].astype(int)

    if df.empty:
        raise ValueError("No valid rows remained after parsing the input CSV.")

    return df


def add_baseline_makespan(df):
    base = df[df["plan"] == "makespan"].copy()

    if base.empty:
        raise ValueError("No rows with plan == 'makespan'. Cannot compute baseline M.")

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
        raise ValueError(f"Found non-positive baseline M for instances: {bad[:10]}")

    out["M"] = out["M"].astype(int)
    out["normalized_cascade"] = out["total_cascade"] / out["M"]

    return out


def instance_level_average(df):
    return (
        df.groupby(["instance", "M", "plan", "k"], as_index=False)
        .agg(
            avg_normalized_cascade=("normalized_cascade", "mean"),
            avg_total_cascade=("total_cascade", "mean"),
            n_executions=("total_cascade", "size"),
        )
    )


def grouped_by_makespan(inst):
    grouped = (
        inst.groupby(["M", "k", "plan"], as_index=False)
        .agg(
            avg_normalized_cascade=("avg_normalized_cascade", "mean"),
            avg_total_cascade=("avg_total_cascade", "mean"),
            instances=("instance", "nunique"),
        )
    )

    grouped["plan_order"] = grouped["plan"].map(
        lambda p: PLAN_ORDER.index(p) if p in PLAN_ORDER else 99
    )
    grouped = grouped.sort_values(["k", "M", "plan_order"]).drop(columns=["plan_order"])

    return grouped


def make_pivot_tables(grouped):
    piv = grouped.pivot_table(
        index=["k", "M"],
        columns="plan",
        values="avg_normalized_cascade",
        aggfunc="first",
    ).reset_index()

    plan_cols = [p for p in PLAN_ORDER if p in piv.columns]
    piv = piv[["k", "M"] + plan_cols]

    counts = (
        grouped[grouped["plan"] == "makespan"][["k", "M", "instances"]]
        .drop_duplicates()
    )
    if not counts.empty:
        piv = piv.merge(counts, on=["k", "M"], how="left")
        cols = ["k", "M", "instances"] + plan_cols
        piv = piv[cols]

    per_k = {}
    for k in sorted(piv["k"].unique()):
        sub = piv[piv["k"] == k].copy()
        sub = sub.drop(columns=["k"])
        per_k[int(k)] = sub

    return piv, per_k


def format_numeric_table(table):
    rounded = table.copy()
    for col in rounded.columns:
        if col not in ["M", "instances"]:
            rounded[col] = rounded[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )
    return rounded


def main():
    parser = argparse.ArgumentParser(description="Generate CSVs for ICAART Tables 4-6.")
    parser.add_argument("--input", required=True, help="Path to adg_delay_details.csv.")
    parser.add_argument("--out-dir", default="grouped_makespan_tables")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = standardize_columns(pd.read_csv(args.input))
    df = df[df["k"].isin([5, 10, 15])]
    if "krobust" in set(df["plan"]):
        df = df[df["plan"].isin(PLAN_ORDER)].copy()
    df = add_baseline_makespan(df)

    grouped = grouped_by_makespan(instance_level_average(df))
    _, per_k = make_pivot_tables(grouped)
    for k, table in per_k.items():
        stem = f"table_grouped_by_makespan_delta{k}"
        table.to_csv(out_dir / f"{stem}_numeric.csv", index=False)
        rounded = format_numeric_table(table)
        rounded.to_csv(out_dir / f"{stem}_rounded.csv", index=False)

    print(f"Saved CSVs for Tables 4-6 in: {out_dir}")


if __name__ == "__main__":
    main()
