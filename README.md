# Multi-Agent Path Finding with Diverse Plans
The experiments use 63 instances drawn from seven maps, with three scenarios per map and two, three, or four agents. They compare a makespan baseline with four plans optimised for diversity at the same makespan (IV, MTD, SumTD, and Leximin) and with k-robust plans. The Action Dependency Graph (ADG) simulation tests delays of 5, 10, and 15 timesteps.

## Repository contents

| Path | Contents |
| --- | --- |
| `experiment.py` | Solves or loads plans and runs the delay simulation. |
| `mapf_instances.py`, `experiments/` | Instance generator and the 63 grid instances. |
| `mapf-makespan.essence`, `metric1.essence`–`metric4.essence`, `krobust.essence` | Models for the makespan baseline, IV, MTD, SumTD, Leximin, and k-robust plans. |
| `results/plans/` | Saved solutions and manifests for replaying the plans. |
| `adg_delay_details.csv` | Delay trials used to generate the figures and tables. |
| `adg_delay_summary_by_makespan.csv`, `adg_delay_same_makespan_presentation.csv` | Per-instance and makespan-group summaries. |
| `analysis/`, `results/figures/`, `results/tables/` | Analysis scripts, figures and tables. |
| `examples/` | Paper’s illustrative examples. |

## 1. Set up Python in VS Code

On Windows, use Ubuntu in WSL and open the folder with **WSL: Reopen Folder in WSL**. See the [VS Code WSL guide](https://code.visualstudio.com/docs/remote/wsl).

## 2. Set up the solver

The models are written in Essence 1.3. Conjure translates them for Savile Row, which calls the solver. The paper uses OR-Tools. Minion is a supported alternative backend. Follow the [Conjure installation guide](https://conjure.readthedocs.io/en/latest/installation.html).

## 3. Run the experiment 

Regenerate the 63 instances with the paper’s seed, then solve them and run the delay simulation:

```bash
python mapf_instances.py --seed 73 --scenarios 3 --output-dir outputs/fresh/experiments
python experiment.py --instances-dir outputs/fresh/experiments --grid 5x5 --solver or-tools --delay-durations 5,10,15 --delay-step-mode all --robustness-values match --conjure-out-dir outputs/fresh/plans --results-dir outputs/fresh/results
```

## 4. Generate the figures and tables

Run these commands from the repository root:

```bash
python analysis/make_precise_adg_plots_v3.py --input adg_delay_details.csv --out-dir outputs/analysis/cascade
python analysis/make_grouped_by_makespan_table.py --input adg_delay_details.csv --out-dir outputs/analysis/grouped
python analysis/make_makespan_comparison_plots.py --input adg_delay_details.csv --out-dir outputs/analysis/makespan
```

The commands generate Tables 3–6 and Figures 5–6 under `outputs/analysis/`. To analyse a fresh solve or replay, replace `adg_delay_details.csv` in each command with the path to that run’s CSV.
