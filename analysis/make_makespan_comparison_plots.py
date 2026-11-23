#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXECUTED_PLANS = ["makespan", "IV", "MTD", "SumTD", "Leximin"]
KROBUST_PLANNED = "k-robust planned"
KROBUST_EXECUTED = "k-robust executed"
METHOD_ORDER = EXECUTED_PLANS + [KROBUST_PLANNED, KROBUST_EXECUTED]

REQUIRED_COLUMNS = {
    "instance",
    "plan",
    "baseline_makespan",
    "planned_makespan",
    "executed_makespan",
    "delay_duration",
    "delayed_agent",
    "delay_timestep",
}

VALID_EXECUTION_FLAGS = {"execution_safe": True, "converged": True, "adg_has_cycle": False}

COLORS = {
    "makespan": "tab:blue",
    "IV": "tab:orange",
    "MTD": "tab:green",
    "SumTD": "tab:red",
    "Leximin": "tab:purple",
    KROBUST_PLANNED: "black",
    KROBUST_EXECUTED: "tab:brown",
}

LINE_STYLES = {
    "makespan": "-",
    "IV": "-",
    "MTD": "-",
    "SumTD": "-",
    "Leximin": "-",
    KROBUST_PLANNED: "--",
    KROBUST_EXECUTED: "-.",
}

DISPLAY_LABELS = {
    "makespan": "makespan",
    "IV": "IV",
    "MTD": "MTD",
    "SumTD": "SumTD",
    "Leximin": "Leximin",
    KROBUST_PLANNED: KROBUST_PLANNED,
    KROBUST_EXECUTED: KROBUST_EXECUTED,
}


def parse_int_set(text: str) -> set[int]:
    values = {int(value.strip()) for value in text.split(",") if value.strip()}
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def boolean_series(series: pd.Series, name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"Column {name} contains invalid booleans: {unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def validate_execution_flags(data: pd.DataFrame, input_path: Path) -> None:
    if not set(VALID_EXECUTION_FLAGS).issubset(data.columns):
        metadata_path = input_path.with_name("run_metadata.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            invalid_count = metadata.get("invalid_adg_rows", 0)
            if invalid_count:
                raise ValueError(f"Run metadata reports {invalid_count} invalid execution rows.")

    invalid = pd.Series(False, index=data.index)
    for column, expected in VALID_EXECUTION_FLAGS.items():
        if column in data:
            data[column] = boolean_series(data[column], column)
            invalid |= data[column] != expected
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} invalid execution rows.")


def require_constant_by_instance(data: pd.DataFrame, column: str) -> None:
    counts = data.groupby("instance")[column].nunique(dropna=False)
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(
            f"{column} is not constant within instances: {list(bad.index[:5])}"
        )


def load_and_validate(
    input_path: Path,
    expected_delays: set[int],
    expected_instances: int,
) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in [
        "baseline_makespan",
        "planned_makespan",
        "executed_makespan",
        "delay_duration",
        "delayed_agent",
        "delay_timestep",
    ]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    data["delay_duration"] = data["delay_duration"].astype(int)

    validate_execution_flags(data, input_path)

    observed_instances = data["instance"].nunique()
    if expected_instances and observed_instances != expected_instances:
        raise ValueError(
            f"Expected {expected_instances} instances, found {observed_instances}."
        )
    observed_delays = set(data["delay_duration"])
    if observed_delays != expected_delays:
        raise ValueError(
            "Delay set mismatch: "
            f"expected={sorted(expected_delays)}, observed={sorted(observed_delays)}"
        )

    expected_plans = set(EXECUTED_PLANS + ["krobust"])
    observed_plans = set(data["plan"])
    if observed_plans != expected_plans:
        raise ValueError(
            "Plan set mismatch: "
            f"missing={sorted(expected_plans-observed_plans)}, "
            f"unexpected={sorted(observed_plans-expected_plans)}"
        )

    require_constant_by_instance(data, "baseline_makespan")
    if (data["baseline_makespan"] <= 0).any():
        raise ValueError("Baseline makespan M must be positive.")

    ordinary = data[data["plan"].isin(EXECUTED_PLANS)]
    if not (
        ordinary["planned_makespan"] == ordinary["baseline_makespan"]
    ).all():
        raise ValueError("An ordinary/diversity plan does not preserve M.")

    robust = data[data["plan"] == "krobust"].copy()
  
    if "robustness_kappa" in robust:
        kappa = pd.to_numeric(robust["robustness_kappa"], errors="raise")
        if not (kappa == robust["delay_duration"]).all():
            raise ValueError("Kappa-robust rows must use matching kappa and delay.")
        data.loc[robust.index, "robustness_kappa"] = kappa
    if (robust["planned_makespan"] < robust["baseline_makespan"]).any():
        raise ValueError("A robust planned makespan MR is below M.")

    completeness = data.groupby(["plan", "delay_duration"])["instance"].nunique()
    incomplete = completeness[completeness != observed_instances]
    if not incomplete.empty:
        raise ValueError(
            "Some plan/delay groups do not contain every instance: "
            f"{incomplete.to_dict()}"
        )

    if (data["executed_makespan"] < data["baseline_makespan"]).any():
        raise ValueError("An executed makespan is below the ordinary optimum M.")

    return data


def comparison_rows(data: pd.DataFrame) -> pd.DataFrame:
    ordinary = data[data["plan"].isin(EXECUTED_PLANS)].copy()
    ordinary["method"] = ordinary["plan"]
    ordinary["comparison_makespan"] = ordinary["executed_makespan"]
    ordinary["quantity"] = "executed makespan"

    robust = data[data["plan"] == "krobust"].copy()
    robust_planned = robust.copy()
    robust_planned["method"] = KROBUST_PLANNED
    robust_planned["comparison_makespan"] = robust_planned["planned_makespan"]
    robust_planned["quantity"] = "planned makespan MR"

    robust_executed = robust.copy()
    robust_executed["method"] = KROBUST_EXECUTED
    robust_executed["comparison_makespan"] = robust_executed["executed_makespan"]
    robust_executed["quantity"] = "executed makespan"

    out = pd.concat(
        [ordinary, robust_planned, robust_executed], ignore_index=True
    )
    out["relative_increase"] = (
        out["comparison_makespan"] - out["baseline_makespan"]
    ) / out["baseline_makespan"]
    return out


def cdf_coordinates(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(pd.to_numeric(values, errors="raise").to_numpy(dtype=float))
    if x.size == 0:
        raise ValueError("Cannot plot CDF with no observations.")
    x = np.concatenate(([x[0]], x))
    y = np.arange(x.size, dtype=float) / (x.size - 1)
    return x, y


def plot_cdf(comparisons: pd.DataFrame, delay: int, output_dir: Path) -> None:
    subset = comparisons[comparisons["delay_duration"] == delay]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for method in METHOD_ORDER:
        values = subset.loc[subset["method"] == method, "relative_increase"]
        if values.empty:
            raise ValueError(f"No execution-level observations for {method}, delay={delay}.")
        x, y = cdf_coordinates(values)
        ax.step(
            x,
            y,
            where="post",
            label=DISPLAY_LABELS[method],
            color=COLORS[method],
            linestyle=LINE_STYLES[method],
            linewidth=2.5,
        )

    ax.axvline(0, color="0.5", linewidth=1.2, linestyle=":")
    ax.set_xlabel("Relative makespan increase over M")
    ax.set_ylabel("Fraction of executions")
    ax.set_title(f"Cumulative distribution of relative makespan increase, delay={delay}")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()

    stub = f"cdf_execution_relative_makespan_delay{delay}"
    fig.savefig(output_dir / f"{stub}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stub}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the execution-level CDFs in Figure 6.")
    parser.add_argument("--input", required=True, help="Path to adg_delay_details.csv.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-delays", default="5,10,15")
    parser.add_argument("--expected-instances", type=int, default=63)
    parser.add_argument("--cdf-level", choices=["execution"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_delays = parse_int_set(args.expected_delays)
    data = load_and_validate(Path(args.input), expected_delays, args.expected_instances)
    comparisons = comparison_rows(data)
    for delay in sorted(expected_delays):
        plot_cdf(comparisons, delay, output_dir)

    print(f"Saved Figure 6 in: {output_dir}")


if __name__ == "__main__":
    main()


