#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


METRICS = ["metric1", "metric2", "metric3", "metric4"]

METRIC_LABEL = {
    "metric1": "IV",
    "metric2": "MTD",
    "metric3": "SumTD",
    "metric4": "Leximin",
}

PLAN_ORDER = ["makespan", "IV", "MTD", "SumTD", "Leximin"]

DEFAULT_DELAY_DURATIONS = [5, 10, 15]
DEFAULT_SOLVER = "or-tools"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_STEP_ATTEMPTS = 50
DEFAULT_MAX_METRIC4_ATTEMPTS = 50


def safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default

def reciprocal_scale_for_steps(steps):
    upper_bound = max(2, int(steps))
    return math.lcm(*range(1, upper_bound))

def natural_key(s):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(s))]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def command_version(command: str) -> str:
    try:
        proc = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if proc.returncode:
        return f"return-code-{proc.returncode}"
    output = proc.stdout.strip().splitlines()
    return output[0] if output else "unavailable"

def required_model_paths(work_dir, robustness_values):
    paths = [work_dir / "mapf-makespan.essence"]
    paths.extend(work_dir / f"{metric}.essence" for metric in METRICS)
    if robustness_values:
        paths.append(work_dir / "krobust.essence")
    return paths

def validate_required_models(work_dir, robustness_values):
    paths = required_model_paths(work_dir, robustness_values)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required Essence model files are missing:\n{formatted}")
    return paths

def repository_relative_path(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return None

def csv_solution_path(solution_file) -> str:
    path = Path(solution_file)
    return repository_relative_path(path) or path.name

def write_run_metadata(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def solution_manifest_path(conjure_out_dir: Path, instance_name: str) -> Path:
    return conjure_out_dir / "solution-manifests" / f"{instance_name}.json"


def save_solution_manifest(param_meta, solution_infos, conjure_out_dir):
    manifest_path = solution_manifest_path(conjure_out_dir, param_meta["name"])
    ensure_dir(manifest_path.parent)

    plans = {}
    plan_directory = conjure_out_dir.resolve()
    for plan_name, info in solution_infos.items():
        solution_file = info.get("solution_file")
        plans[plan_name] = {
            "plan_label": info.get("plan_label", plan_name),
            "solution_file": (
                Path(solution_file).resolve().relative_to(plan_directory).as_posix()
                if solution_file else ""
            ),
            "optimality_proven": bool(info.get("optimality_proven", False)),
            "used_fallback_incumbent": bool(
                info.get("used_fallback_incumbent", False)
            ),
            "fallback_reason": info.get("fallback_reason", ""),
            "robustness_kappa": info.get("robustness_kappa", ""),
            "model_name": info.get("model_name", ""),
            "model_sha256": info.get("model_sha256", ""),
        }

    payload = {
        "schema_version": 3,
        "instance": param_meta["name"],
        "plans": plans,
    }

    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)


def load_solution_manifest(param_meta, conjure_out_dir):
    manifest_path = solution_manifest_path(conjure_out_dir, param_meta["name"])
    if not manifest_path.exists():
        return {}

    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"    WARNING: cannot read solution manifest {manifest_path}: {exc}")
        return {}

    version = payload.get("schema_version")
    if version not in (2, 3):
        print(f"    WARNING: unsupported solution manifest version in {manifest_path}")
        return {}
    if payload.get("instance") != param_meta["name"]:
        print(f"    WARNING: solution manifest instance mismatch in {manifest_path}")
        return {}

    plans = payload.get("plans", {})
    if not isinstance(plans, dict):
        return {}
    if version == 3:
        for entry in plans.values():
            if isinstance(entry, dict) and entry.get("solution_file"):
                entry["solution_file"] = conjure_out_dir / entry["solution_file"]
    return plans


def manifest_solution_file(manifest_entry):
    if not manifest_entry:
        return None
    value = manifest_entry.get("solution_file", "")
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def manifest_matches_model(manifest_entry, model_path: Path) -> bool:
    recorded_hash = manifest_entry.get("model_sha256", "")
    recorded_name = manifest_entry.get("model_name", "")
    return bool(
        recorded_hash
        and recorded_name == model_path.name
        and model_path.is_file()
        and recorded_hash == sha256_file(model_path)
    )

def command_exists(command: str) -> bool:
    return shutil.which(command) is not None

def strip_given_from_param(param_text: str, name: str) -> str:
    return re.sub(
        rf"(?m)^letting\s+{re.escape(name)}\s+be\s+.*(?:\n|$)",
        "",
        param_text,
    )


def replace_steps(param_text: str, steps: int) -> str:
    stripped = strip_given_from_param(param_text, "steps")
    stripped = re.sub(
        r"(?ms)^letting\s+T_incumbent\s+be\s+\{.*?\}\s*$",
        "",
        stripped,
    )
    return stripped.rstrip() + f"\n\nletting steps be {steps}\n"


def parse_map_from_param(param_text: str):
    block = extract_balanced_after(param_text, "letting map be")
    if not block:
        return []

    rows = []
    for row_text in re.findall(r"\[([^\[\]]+)\]", block):
        nums = [int(x) for x in re.findall(r"-?\d+", row_text)]
        if nums:
            rows.append(nums)

    return rows


def parse_agents_from_param(param_text: str):
    block = extract_balanced_after(param_text, "letting agents be")
    if not block:
        return []

    agents = []

    for m in re.finditer(
        r"record\s*\{\s*"
        r"start_x\s*=\s*(\d+)\s*,\s*"
        r"start_y\s*=\s*(\d+)\s*,\s*"
        r"goal_x\s*=\s*(\d+)\s*,\s*"
        r"goal_y\s*=\s*(\d+)\s*"
        r"\}",
        block,
        flags=re.DOTALL,
    ):
        agents.append({
            "start": (int(m.group(1)), int(m.group(2))),
            "goal": (int(m.group(3)), int(m.group(4))),
        })

    return agents


def bfs_shortest_distance(grid, start, goal):
    if not grid:
        return None

    h = len(grid)
    w = len(grid[0])

    sr, sc = start
    gr, gc = goal

    sr -= 1
    sc -= 1
    gr -= 1
    gc -= 1

    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None

    if grid[sr][sc] != 0 or grid[gr][gc] != 0:
        return None

    q = deque([(sr, sc, 0)])
    seen = {(sr, sc)}

    while q:
        r, c, d = q.popleft()

        if (r, c) == (gr, gc):
            return d

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc

            if (
                0 <= nr < h
                and 0 <= nc < w
                and grid[nr][nc] == 0
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))

    return None


def compute_steps_lower_bound_from_param(param_text: str):
    grid = parse_map_from_param(param_text)
    agents = parse_agents_from_param(param_text)

    if not grid or not agents:
        return None

    distances = []

    for ag in agents:
        d = bfs_shortest_distance(grid, ag["start"], ag["goal"])
        if d is None:
            return None
        distances.append(d)

    return max(distances) + 1


def remove_old_solution_files(output_dir: Path, param_stem: str):
    for sol in output_dir.glob(f"*-{param_stem}-solution*.solution"):
        try:
            sol.unlink()
        except FileNotFoundError:
            pass


def collect_params(instances_dir: Path, grid_filter=None, agents_filter=None):
    params_dir = instances_dir / "params"

    if not params_dir.exists():
        raise FileNotFoundError(f"Missing params directory: {params_dir}")

    all_params = []

    grid_dirs = [params_dir / grid_filter] if grid_filter else [
        d for d in params_dir.iterdir() if d.is_dir()
    ]

    for grid_dir in sorted(grid_dirs, key=natural_key):
        if not grid_dir.exists():
            print(f"WARNING: grid directory not found: {grid_dir}")
            continue

        for param_path in sorted(grid_dir.rglob("*.param"), key=lambda p: natural_key(p.name)):
            stem = param_path.stem

            m = re.match(r"(\d+x\d+)_(\d+)pct_scen(\d+)_agents(\d+)$", stem)
            if not m:
                print(f"WARNING: ignoring param with unexpected name: {param_path}")
                continue

            n_agents = int(m.group(4))
            if agents_filter is not None and n_agents != agents_filter:
                continue

            all_params.append({
                "name": stem,
                "grid": m.group(1),
                "obs_pct": int(m.group(2)),
                "scenario": int(m.group(3)),
                "n_agents": n_agents,
                "param_path": param_path,
            })

    return all_params


def run_conjure(
    essence_file: Path,
    param_file: Path,
    output_dir: Path,
    work_dir: Path,
    solver: str,
    timeout: int,
):
    cmd = [
        "conjure",
        "solve",
        str(essence_file),
        str(param_file),
        "--copy-solutions=no",
        "--solver",
        solver,
        "--line-width",
        "100000",
        "-o",
        str(output_dir),
    ]

    try:
        start = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        return proc.returncode, proc.stdout, proc.stderr, elapsed, cmd

    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", e.stderr or "TIMEOUT", timeout, cmd

    except FileNotFoundError:
        return -2, "", "CONJURE_NOT_FOUND", 0.0, cmd


def print_conjure_error(prefix, rc, out, err, cmd):
    print(f"{prefix} Conjure failed with return code {rc}")
    print(f"{prefix} Command:")
    print(" ".join(str(x) for x in cmd))

    if out and out.strip():
        print(f"{prefix} STDOUT tail:")
        print(out[-2500:])

    if err and err.strip():
        print(f"{prefix} STDERR tail:")
        print(err[-2500:])


def find_solution_file(output_dir: Path, param_stem: str):
    candidates = sorted(
        output_dir.glob(f"*-{param_stem}-solution*.solution"),
        key=lambda p: (p.stat().st_mtime, natural_key(p.name)),
    )

    if candidates:
        return candidates[-1]

    candidates = sorted(
        output_dir.glob(f"*{param_stem}*solution*.solution"),
        key=lambda p: (p.stat().st_mtime, natural_key(p.name)),
    )
    return candidates[-1] if candidates else None


def extract_balanced_after(text: str, marker: str):
    idx = text.find(marker)
    if idx < 0:
        return ""

    start = text.find("[", idx)
    if start < 0:
        return ""

    depth = 0
    for pos in range(start, len(text)):
        ch = text[pos]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:pos + 1]

    return ""


def parse_agent_location(solution_file: Path):
    text = solution_file.read_text(errors="replace")

    block = extract_balanced_after(text, "letting agent_location be")
    if not block:
        return []

    agent_blocks = re.findall(
        r"\[((?:\s*record\s+\{[^}]+\}\s*,?)+)\s*;\s*int\(1\.\.\d+\)\s*\]",
        block,
        flags=re.DOTALL,
    )

    paths = []
    for agent_block in agent_blocks:
        coords = re.findall(
            r"record\s+\{\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\}",
            agent_block,
        )
        if coords:
            paths.append([(int(x), int(y)) for x, y in coords])

    if not paths:
        coords = re.findall(
            r"record\s+\{\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\}",
            block,
        )
        if coords:
            print(f"WARNING: found coordinates but could not split agents in {solution_file}")

    return paths


def parse_solution_metrics(solution_file: Path):
    text = solution_file.read_text(errors="replace")
    metrics = {}

    m = re.search(r"letting\s+IV\s+be\s+(-?\d+)", text)
    if m:
        metrics["IV"] = int(m.group(1))

    m = re.search(r"letting\s+T_minimum\s+be\s+(-?\d+)", text)
    if m:
        metrics["MTD"] = int(m.group(1))

    m = re.search(r"letting\s+T_recip\s+be\s+(-?\d+)", text)
    if m:
        metrics["SumTD_raw"] = int(m.group(1))

    m = re.search(r"letting\s+T_counts\s+be\s+\[([^\]]+)\]", text, flags=re.DOTALL)
    if m:
        counts_part = m.group(1).split(";")[0]
        counts = [int(x) for x in re.findall(r"-?\d+", counts_part)]
        finite_values = []
        for distance, count in enumerate(counts, start=1):
            finite_values.extend([distance] * count)
        metrics["Leximin_counts"] = tuple(counts)
        metrics["Leximin"] = tuple(finite_values)

    return metrics


def parse_T_text(solution_file: Path):
    text = solution_file.read_text(errors="replace")

    m = re.search(r"(?m)^letting\s+T\s+be\s+(.+)$", text)
    if m:
        return m.group(1).strip()

    idx = text.find("letting T be")
    if idx < 0:
        return ""

    start = text.find("[", idx)
    if start < 0:
        return ""

    depth = 0
    for pos in range(start, len(text)):
        ch = text[pos]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:pos + 1].strip()

    return ""


def parse_solution(solution_file: Path):
    paths = parse_agent_location(solution_file)
    metrics = parse_solution_metrics(solution_file)
    if paths and "SumTD_raw" in metrics:
        steps = max(len(path) for path in paths)
        scale = reciprocal_scale_for_steps(steps)
        metrics["SumTD_scale"] = scale
        metrics["SumTD"] = round(metrics["SumTD_raw"] / scale, 12)
    T_text = parse_T_text(solution_file)
    return paths, metrics, T_text


def compute_diversity(paths):
    if not paths:
        return {
            "IV": "",
            "MTD": "",
            "MTD_is_top": "",
            "MTD_single_delay_tolerance": "",
            "SumTD": "",
            "Leximin": "",
        }

    vertex_to_agents = defaultdict(set)
    for idx, path in enumerate(paths, start=1):
        for v in set(path):
            vertex_to_agents[v].add(idx)

    IV = sum(len(agent_set) for agent_set in vertex_to_agents.values() if len(agent_set) > 1)

    diffs = []
    n = len(paths)

    pairs = ((i, j) for i in range(n) for j in range(i + 1, n))

    for i, j in pairs:
        pos_times_j = defaultdict(list)
        for t2, pos in enumerate(paths[j]):
            pos_times_j[pos].append(t2)

        for t1, pos in enumerate(paths[i]):
            if pos in pos_times_j:
                d = min(abs(t1 - t2) for t2 in pos_times_j[pos])
                if d > 0:
                    diffs.append(d)

    if diffs:
        MTD = min(diffs)
        MTD_is_top = False
        SumTD = round(sum(1.0 / d for d in diffs if d > 0), 12)
        Leximin = tuple(sorted(diffs))
    else:
        MTD = max(len(path) for path in paths)
        MTD_is_top = True
        SumTD = 0.0
        Leximin = tuple()

    return {
        "IV": IV,
        "MTD": MTD,
        "MTD_is_top": MTD_is_top,
        "MTD_single_delay_tolerance": "" if MTD_is_top else max(0, MTD - 1),
        "SumTD": SumTD,
        "Leximin": Leximin,
    }


def paths_to_agent_dict(paths):
    return {i + 1: list(path) for i, path in enumerate(paths)}


def truncate_goal_waits(paths_dict):
    out = {}

    for a, path in paths_dict.items():
        if not path:
            out[a] = []
            continue

        goal = path[-1]
        end = len(path) - 1

        while end > 0 and path[end - 1] == goal:
            end -= 1

        out[a] = path[:end + 1]

    return out


def first_goal_arrival_index(paths_dict):
    out = {}
    for a, path in paths_dict.items():
        if not path:
            out[a] = 0
            continue
        goal = path[-1]
        end = len(path) - 1
        while end > 0 and path[end - 1] == goal:
            end -= 1
        out[a] = end
    return out


def build_adg_edges(paths):
    vertex_visits = defaultdict(list)

    for agent, path in paths.items():
        t = 0
        while t < len(path):
            v = path[t]
            arrive_t = t
            depart_t = t

            while depart_t + 1 < len(path) and path[depart_t + 1] == v:
                depart_t += 1

            vertex_visits[v].append((arrive_t, depart_t, agent))
            t = depart_t + 1

    edges = []

    for visits in vertex_visits.values():
        if len({agent for _, _, agent in visits}) <= 1:
            continue

        visits.sort(key=lambda x: (x[0], x[1], x[2]))

        for i in range(len(visits) - 1):
            prev_arrive, prev_depart, prev_agent = visits[i]
            curr_arrive, curr_depart, curr_agent = visits[i + 1]

            if prev_agent != curr_agent:
                if prev_depart + 1 < len(paths[prev_agent]):
                    predecessor_state = prev_depart + 1
                    lag = 0
                else:
                    predecessor_state = prev_depart
                    lag = 1
                edges.append(
                    (prev_agent, predecessor_state, curr_agent, curr_arrive, lag)
                )

    return edges


def has_adg_cycle(num_agents, paths, edges):
    nodes = set()
    weighted_edges = []

    for a, path in paths.items():
        for t in range(len(path)):
            nodes.add((a, t))
            if t + 1 < len(path):
                weighted_edges.append(((a, t), (a, t + 1), 1))

    for pa, pt, ca, ct, lag in edges:
        weighted_edges.append(((pa, pt), (ca, ct), lag))
        nodes.add((pa, pt))
        nodes.add((ca, ct))

    distance = {node: 0 for node in nodes}
    for pass_index in range(len(nodes)):
        changed = False
        for source, target, lag in weighted_edges:
            candidate = distance[source] + lag
            if distance[target] < candidate:
                distance[target] = candidate
                changed = True
        if not changed:
            return False
        if pass_index == len(nodes) - 1:
            return True
    return False


def validate_execution_safety(paths, exec_time, disappear_at_goal=False):
    if not paths:
        return {"safe": True, "conflict_type": "", "conflict_time": ""}

    def position_at(agent, clock):
        path = paths[agent]
        times = exec_time[agent]
        if not path:
            return None
        if clock < times[0]:
            return path[0]

        state = -1
        for index, arrival in enumerate(times):
            if arrival <= clock:
                state = index
            else:
                break

        if state < 0:
            return path[0]
        if state == len(path) - 1 and disappear_at_goal and clock > times[-1]:
            return None
        return path[state]

    agents = sorted(paths)
    end_time = max(times[-1] for times in exec_time.values())
    previous = {}

    for clock in range(0, end_time + 1):
        current = {agent: position_at(agent, clock) for agent in agents}

        occupied = defaultdict(list)
        for agent, vertex in current.items():
            if vertex is not None:
                occupied[vertex].append(agent)
        for vertex, visitors in occupied.items():
            if len(visitors) > 1:
                return {
                    "safe": False,
                    "conflict_type": "vertex",
                    "conflict_time": clock,
                    "conflict_agents": tuple(visitors),
                    "conflict_vertex": vertex,
                }

        if clock > 0:
            for i, first in enumerate(agents):
                for second in agents[i + 1:]:
                    f0, f1 = previous.get(first), current[first]
                    s0, s1 = previous.get(second), current[second]
                    if (
                        None not in (f0, f1, s0, s1)
                        and f0 != f1
                        and f0 == s1
                        and s0 == f1
                    ):
                        return {
                            "safe": False,
                            "conflict_type": "edge_swap",
                            "conflict_time": clock,
                            "conflict_agents": (first, second),
                            "conflict_vertex": (f0, f1),
                        }
        previous = current

    return {"safe": True, "conflict_type": "", "conflict_time": ""}


def simulate_adg_systematic(
    paths_list, delay_agent, delay_timestep, delay_duration
):
    raw = paths_to_agent_dict(paths_list)
    paths = raw  
    agents = sorted(paths.keys())
    arrival_index = first_goal_arrival_index(raw)

    if delay_agent not in paths:
        raise ValueError(f"delay_agent {delay_agent} not in paths")
    if delay_duration < 0:
        raise ValueError("delay_duration must be non-negative")

    if delay_timestep < 0 or delay_timestep >= len(paths[delay_agent]):
        raise ValueError(
            f"delay_timestep {delay_timestep} invalid for agent {delay_agent}"
        )

    edges = build_adg_edges(paths)
    cycle = has_adg_cycle(len(agents), paths, edges)
    if cycle:
        raise RuntimeError("ADG contains an infeasible positive-lag cycle")

    constraints = defaultdict(list)
    for prev_agent, prev_t, curr_agent, curr_t, lag in edges:
        constraints[(curr_agent, curr_t)].append((prev_agent, prev_t, lag))

    exec_time = {
        a: list(range(len(paths[a])))
        for a in agents
    }

    for t in range(delay_timestep, len(paths[delay_agent])):
        exec_time[delay_agent][t] += delay_duration

    max_iterations = max(1000, 20 * sum(len(paths[a]) for a in agents))
    converged = False

    for _ in range(max_iterations):
        changed = False

        for a in agents:
            for t in range(len(paths[a])):
                earliest = 0

                if t > 0:
                    earliest = max(earliest, exec_time[a][t - 1] + 1)

                for prev_agent, prev_t, lag in constraints[(a, t)]:
                    earliest = max(earliest, exec_time[prev_agent][prev_t] + lag)

                if exec_time[a][t] < earliest:
                    exec_time[a][t] = earliest
                    changed = True

        if not changed:
            converged = True
            break

    if not converged:
        raise RuntimeError("ADG propagation did not converge")

    safety = validate_execution_safety(paths, exec_time, disappear_at_goal=False)
    if not safety["safe"]:
        raise RuntimeError(
            "Unsafe ADG execution: "
            f"{safety['conflict_type']} conflict at time {safety['conflict_time']}"
        )

    planned_arrival = {a: arrival_index[a] for a in agents}
    executed_arrival = {a: exec_time[a][arrival_index[a]] for a in agents}

    planned_makespan = max(planned_arrival.values())
    executed_makespan = max(executed_arrival.values())

    total_delay_by_agent = {
        a: max(0, executed_arrival[a] - planned_arrival[a])
        for a in agents
    }

    cascade_by_agent = {}
    for a in agents:
        if a == delay_agent:
            cascade_by_agent[a] = max(
                0, total_delay_by_agent[a] - delay_duration
            )
        else:
            cascade_by_agent[a] = total_delay_by_agent[a]

    total_cascade_others = sum(
        cascade_by_agent[a]
        for a in agents
        if a != delay_agent
    )

    total_cascade_all = sum(cascade_by_agent.values())

    affected_agents_others = sum(
        1 for a in agents
        if a != delay_agent and cascade_by_agent[a] > 0
    )

    return {
        "planned_makespan": planned_makespan,
        "executed_makespan": executed_makespan,
        "makespan_increase": executed_makespan - planned_makespan,
        "total_delay_by_agent": total_delay_by_agent,
        "cascade_by_agent": cascade_by_agent,
        "total_cascade": total_cascade_others,
        "total_cascade_all": total_cascade_all,
        "affected_agents": affected_agents_others,
        "n_adg_edges": len(edges),
        "adg_has_cycle": cycle,
        "converged": converged,
        "execution_safe": safety["safe"],
        "conflict_type": safety["conflict_type"],
        "conflict_time": safety["conflict_time"],
        "exec_time": exec_time,
        "execution_policy": "adg",
    }


def simulate_krobust_systematic(
    paths_list,
    delay_agent,
    delay_timestep,
    delay_duration,
    robustness_kappa,
):
    if delay_duration > robustness_kappa:
        raise ValueError(
            f"delta={delay_duration} exceeds robustness kappa={robustness_kappa}"
        )
    raw = paths_to_agent_dict(paths_list)
    paths = raw  
    arrival_index = first_goal_arrival_index(raw)
    if delay_agent not in paths:
        raise ValueError(f"delay_agent {delay_agent} not in paths")
    if delay_timestep < 0 or delay_timestep >= len(paths[delay_agent]):
        raise ValueError(
            f"delay_timestep {delay_timestep} invalid for agent {delay_agent}"
        )

    exec_time = {agent: list(range(len(path))) for agent, path in paths.items()}
    for state in range(delay_timestep, len(paths[delay_agent])):
        exec_time[delay_agent][state] += delay_duration

    safety = validate_execution_safety(paths, exec_time, disappear_at_goal=False)
    if not safety["safe"]:
        raise RuntimeError(
            f"kappa-robustness violation: {safety['conflict_type']} conflict "
            f"at time {safety['conflict_time']} for delta={delay_duration}, "
            f"kappa={robustness_kappa}"
        )

    planned_arrival = {agent: arrival_index[agent] for agent in paths}
    executed_arrival = {
        agent: exec_time[agent][arrival_index[agent]] for agent in sorted(paths)
    }
    planned_makespan = max(planned_arrival.values())
    executed_makespan = max(executed_arrival.values())
    total_delay_by_agent = {
        agent: executed_arrival[agent] - planned_arrival[agent]
        for agent in sorted(paths)
    }
    cascade_by_agent = {agent: 0 for agent in sorted(paths)}

    return {
        "planned_makespan": planned_makespan,
        "executed_makespan": executed_makespan,
        "makespan_increase": executed_makespan - planned_makespan,
        "total_delay_by_agent": total_delay_by_agent,
        "cascade_by_agent": cascade_by_agent,
        "total_cascade": 0,
        "total_cascade_all": 0,
        "affected_agents": 0,
        "n_adg_edges": 0,
        "adg_has_cycle": False,
        "converged": True,
        "execution_safe": True,
        "conflict_type": "",
        "conflict_time": "",
        "exec_time": exec_time,
        "execution_policy": "blind-krobust",
    }


def solve_makespan_once(
    param_meta,
    param_text,
    work_dir,
    conjure_out_dir,
    solver,
    timeout,
    max_step_attempts,
):
    name = param_meta["name"]

    m = re.search(r"letting\s+steps\s+be\s+(\d+)", param_text)
    if not m:
        print(f"  SKIP {name}: no steps in parameter file")
        return None

    param_steps = int(m.group(1))
    lower_bound_steps = compute_steps_lower_bound_from_param(param_text)

    if lower_bound_steps is None:
        initial_steps = param_steps
        print(
            f"    WARNING: could not compute shortest-path lower bound; "
            f"starting from param steps={param_steps}"
        )
    else:
        initial_steps = lower_bound_steps
        if param_steps != lower_bound_steps:
            print(
                f"    makespan search starts from BFS lower bound steps={lower_bound_steps} "
                f"(param file had steps={param_steps})"
            )

    mks_dir = conjure_out_dir / "makespan"
    ensure_dir(mks_dir)
    model_path = work_dir / "mapf-makespan.essence"

    for attempt in range(1, max_step_attempts + 1):
        steps = initial_steps + attempt - 1
        attempt_tag = f"{attempt:03d}"
        param_stem = f"{name}_{attempt_tag}"
        param_file = mks_dir / f"{param_stem}.param"

        param_file.write_text(replace_steps(param_text, steps))
        remove_old_solution_files(mks_dir, param_stem)

        rc, out, err, elapsed, cmd = run_conjure(
            model_path,
            param_file,
            mks_dir,
            work_dir,
            solver,
            timeout,
        )

        sol = find_solution_file(mks_dir, param_stem)

        if sol and rc == 0:
            paths, metrics, T_text = parse_solution(sol)
            if paths:
                div = compute_diversity(paths)
                print(
                    f"    makespan found: M={steps - 1} moves "
                    f"({steps} states), attempt={attempt}, time={elapsed:.2f}s"
                )
                return {
                    "solution_file": sol,
                    "paths": paths,
                    "metrics": metrics,
                    "computed_metrics": div,
                    "T_text": T_text,
                    "steps": steps,
                    "attempt": attempt,
                    "elapsed": elapsed,
                    "optimality_proven": True,
                    "model_name": model_path.name,
                    "model_sha256": sha256_file(model_path),
                }

        if rc == -1:
            print(f"    makespan timeout at steps={steps}")
            return None

        if rc != 0:
            print_conjure_error("    [makespan]", rc, out, err, cmd)
            return None


    print(f"    makespan phase failed after {max_step_attempts} attempts")
    return None


def solve_metric(
    metric,
    param_meta,
    param_text,
    makespan_info,
    work_dir,
    conjure_out_dir,
    solver,
    timeout,
    max_step_attempts,
    max_metric4_attempts,
):
    name = param_meta["name"]
    model_name = f"{metric}.essence"
    metric_dir = conjure_out_dir / Path(model_name).stem / "opt"
    ensure_dir(metric_dir)

    opt_steps = makespan_info["steps"]

    if metric == "metric4":
        best = dict(makespan_info)
        best["metrics"] = {}
        best["used_fallback_incumbent"] = True
        best["fallback_reason"] = "initial_makespan_incumbent"
        best["optimality_proven"] = False
        best["model_name"] = model_name
        best["model_sha256"] = sha256_file(work_dir / model_name)
        T_incumbent = makespan_info.get("T_text", "")
        limit = max_metric4_attempts
    else:
        best = None
        T_incumbent = ""
        limit = max_step_attempts

    for attempt in range(1, limit + 1):
        attempt_tag = f"{attempt:03d}"
        param_stem = f"{name}_{attempt_tag}"
        param_file = metric_dir / f"{param_stem}.param"

        content = replace_steps(param_text, opt_steps)

        if metric == "metric3":
            content = strip_given_from_param(content, "reciprocal_scale")
            scale = reciprocal_scale_for_steps(opt_steps)
            content += f"\nletting reciprocal_scale be {scale}\n"

        if metric == "metric4":
            if not T_incumbent:
                print("    Leximin: missing T incumbent; cannot certify optimisation")
                return None
            content += f"\nletting T_incumbent be {{{T_incumbent}}}\n"

        param_file.write_text(content)
        remove_old_solution_files(metric_dir, param_stem)

        rc, out, err, elapsed, cmd = run_conjure(
            work_dir / model_name,
            param_file,
            metric_dir,
            work_dir,
            solver,
            timeout,
        )

        sol = find_solution_file(metric_dir, param_stem)

        if sol:
            if rc != 0:
                print_conjure_error(f"    [{metric}]", rc, out, err, cmd)
                print(
                    f"    {METRIC_LABEL[metric]}: an incumbent exists, but the "
                    "solver did not finish; excluding the uncertified plan"
                )
                return None

            paths, metrics, new_T = parse_solution(sol)

            if not paths:
                print(f"    {METRIC_LABEL[metric]}: solution found but agent_location could not be parsed")
                break

            div = compute_diversity(paths)

            best = {
                "solution_file": sol,
                "paths": paths,
                "metrics": metrics,
                "computed_metrics": div,
                "T_text": new_T,
                "steps": opt_steps,
                "attempt": attempt,
                "elapsed": elapsed,
                "used_fallback_incumbent": False,
                "fallback_reason": "",
                "optimality_proven": metric != "metric4",
                "model_name": model_name,
                "model_sha256": sha256_file(work_dir / model_name),
            }

            print(
                f"    {METRIC_LABEL[metric]} solved: "
                f"attempt={attempt}, time={elapsed:.2f}s"
            )

            if metric == "metric4":
                if not new_T:
                    print("    Leximin: solution has no parseable T; cannot certify optimisation")
                    return None

                if new_T == T_incumbent:
                    print("    Leximin: T did not change after a strict-improvement solve")
                    return None

                T_incumbent = new_T
                continue

            break

        if metric == "metric4":
            if rc != 0:
                print_conjure_error("    [metric4]", rc, out, err, cmd)
                print("    Leximin: no optimality certificate; excluding this plan")
                return None

            best["optimality_proven"] = True
            if best is not None and best.get("used_fallback_incumbent", False):
                best["fallback_reason"] = "certified_no_strict_improvement"
                print("    Leximin: makespan incumbent is certified optimal")
            else:
                print("    Leximin: last incumbent is certified optimal")
            break

        if rc != 0:
            print_conjure_error(f"    [{metric}]", rc, out, err, cmd)

        break

    if metric == "metric4" and best is not None and not best.get("optimality_proven"):
        print(
            f"    Leximin: reached {limit} improvement attempts without an "
            "UNSAT certificate; excluding the uncertified plan"
        )
        return None

    return best


def solve_krobust(
    param_meta,
    param_text,
    makespan_info,
    robustness_kappa,
    work_dir,
    conjure_out_dir,
    solver,
    timeout,
    max_robust_attempts,
):
    name = param_meta["name"]
    robust_dir = conjure_out_dir / f"krobust-kappa{robustness_kappa}"
    ensure_dir(robust_dir)
    model_path = work_dir / "krobust.essence"

    initial_steps = makespan_info["steps"]
    for attempt in range(1, max_robust_attempts + 1):
        steps = initial_steps + attempt - 1
        param_stem = f"{name}_{attempt:03d}"
        param_file = robust_dir / f"{param_stem}.param"
        content = strip_given_from_param(param_text, "robustness_k")
        content = strip_given_from_param(content, "robustness_kappa")
        content = replace_steps(content, steps)
        content += f"\nletting robustness_kappa be {robustness_kappa}\n"
        param_file.write_text(content)
        remove_old_solution_files(robust_dir, param_stem)

        rc, out, err, elapsed, cmd = run_conjure(
            model_path,
            param_file,
            robust_dir,
            work_dir,
            solver,
            timeout,
        )
        sol = find_solution_file(robust_dir, param_stem)

        if sol and rc == 0:
            paths, metrics, T_text = parse_solution(sol)
            if not paths:
                print(f"    kappa-robust {robustness_kappa}: unparseable solution")
                return None
            print(
                f"    kappa-robust {robustness_kappa}: MR={steps - 1} moves, "
                f"attempt={attempt}, time={elapsed:.2f}s"
            )
            return {
                "plan_label": "krobust",
                "solution_file": sol,
                "paths": paths,
                "metrics": metrics,
                "computed_metrics": compute_diversity(paths),
                "T_text": T_text,
                "steps": steps,
                "attempt": attempt,
                "elapsed": elapsed,
                "optimality_proven": True,
                "robustness_kappa": robustness_kappa,
                "model_name": model_path.name,
                "model_sha256": sha256_file(model_path),
            }

        if rc == -1:
            print(f"    kappa-robust {robustness_kappa}: timeout at {steps} states")
            return None
        if rc != 0:
            print_conjure_error(f"    [krobust kappa={robustness_kappa}]", rc, out, err, cmd)
            return None

    print(
        f"    kappa-robust {robustness_kappa}: no plan after "
        f"{max_robust_attempts} horizons"
    )
    return None


def run_solver_for_instance(
    param_meta,
    work_dir,
    conjure_out_dir,
    solver,
    timeout,
    max_step_attempts,
    max_metric4_attempts,
    robustness_values,
    max_robust_attempts,
):
    name = param_meta["name"]
    param_text = param_meta["param_path"].read_text()

    print(f"\nSolving {name}")

    makespan = solve_makespan_once(
        param_meta,
        param_text,
        work_dir,
        conjure_out_dir,
        solver,
        timeout,
        max_step_attempts,
    )

    if not makespan:
        print(f"    FAILED: no makespan solution for {name}")
        return {}

    result = {"makespan": makespan}

    for metric in METRICS:
        metric_result = solve_metric(
            metric,
            param_meta,
            param_text,
            makespan,
            work_dir,
            conjure_out_dir,
            solver,
            timeout,
            max_step_attempts,
            max_metric4_attempts,
        )

        if metric_result:
            result[METRIC_LABEL[metric]] = metric_result
        else:
            print(f"    WARNING: no solution for {METRIC_LABEL[metric]}")

    for robustness_kappa in robustness_values:
        robust_result = solve_krobust(
            param_meta,
            param_text,
            makespan,
            robustness_kappa,
            work_dir,
            conjure_out_dir,
            solver,
            timeout,
            max_robust_attempts,
        )
        if robust_result:
            result[f"krobust_kappa{robustness_kappa}"] = robust_result
        else:
            print(f"    WARNING: no certified kappa-robust plan for kappa={robustness_kappa}")

    return result


def extract_instance_attempt_from_solution(sol_path: Path):
    stem = sol_path.stem
    m = re.search(r"model\d+-(.+)-solution\d+", stem)

    if not m:
        return stem, 0

    middle = m.group(1)
    m2 = re.search(r"(.+)_(\d{3})$", middle)

    if m2:
        return m2.group(1), int(m2.group(2))

    return middle, 0


def load_saved_plan(manifest_entry):
    solution_file = manifest_solution_file(manifest_entry)
    if solution_file is None:
        return None
    paths = parse_agent_location(solution_file)
    if not paths:
        return None

    plan = {"solution_file": solution_file, "paths": paths}
    for key in ("plan_label", "robustness_kappa"):
        if key in manifest_entry:
            plan[key] = manifest_entry[key]
    return plan


def load_existing_solutions_for_instance(
    param_meta,
    conjure_out_dir,
    work_dir,
    robustness_values=(),
):
    name = param_meta["name"]
    manifest = load_solution_manifest(param_meta, conjure_out_dir)
    if not manifest:
        print(f"    SKIP {name}: parse-only mode requires a compatible solution manifest")
        return {}

    result = {}
    if (conjure_out_dir / "makespan").exists():
        entry = manifest.get("makespan", {})
        if not manifest_matches_model(entry, work_dir / "mapf-makespan.essence"):
            print(f"    SKIP {name} makespan: model hash differs from manifest")
            return result
        plan = load_saved_plan(entry)
        if plan:
            result["makespan"] = plan

    for metric in METRICS:
        label = METRIC_LABEL[metric]
        entry = manifest.get(label, {})
        if not manifest_matches_model(entry, work_dir / f"{metric}.essence"):
            print(f"    SKIP {name} {label}: model hash differs from manifest")
            continue
        if not (conjure_out_dir / metric / "opt").exists():
            continue
        plan = load_saved_plan(entry)
        if plan:
            result[label] = plan

    leximin_manifest = manifest.get("Leximin", {})
    certified_fallback = (
        bool(leximin_manifest.get("optimality_proven", False))
        and bool(leximin_manifest.get("used_fallback_incumbent", False))
        and leximin_manifest.get("fallback_reason") == "certified_no_strict_improvement"
        and manifest_matches_model(leximin_manifest, work_dir / "metric4.essence")
    )
    if "makespan" in result and certified_fallback:
        result["Leximin"] = {**result["makespan"], "plan_label": "Leximin"}

    for robustness_kappa in robustness_values:
        if not (conjure_out_dir / f"krobust-kappa{robustness_kappa}").exists():
            continue
        robust_key = f"krobust_kappa{robustness_kappa}"
        entry = manifest.get(robust_key, {})
        if not manifest_matches_model(entry, work_dir / "krobust.essence"):
            print(f"    SKIP {name} krobust {robustness_kappa}: model hash differs from manifest")
            continue
        plan = load_saved_plan(entry)
        if plan:
            plan.setdefault("plan_label", "krobust")
            plan.setdefault("robustness_kappa", robustness_kappa)
            result[robust_key] = plan

    return result


def make_solution_summary_row(param_meta, plan_name, info):
    return {
        "instance": param_meta["name"],
        "grid": param_meta["grid"],
        "obs_pct": param_meta["obs_pct"],
        "scenario": param_meta["scenario"],
        "n_agents": param_meta["n_agents"],
        "plan": plan_name,
        "robustness_kappa": info.get("robustness_kappa", ""),
        "solution_file": csv_solution_path(info["solution_file"]),
        "solve_attempt": info.get("attempt", ""),
        "solve_elapsed_seconds": info.get("elapsed", ""),
    }


def build_paired_injection_points(solution_infos, include_goal=True):
    path_sets = [
        truncate_goal_waits(paths_to_agent_dict(info["paths"]))
        for info in solution_infos
        if info.get("paths")
    ]
    if not path_sets:
        return []

    common_agents = set(path_sets[0])
    for path_set in path_sets[1:]:
        common_agents.intersection_update(path_set)

    points = []
    for agent in sorted(common_agents):
        common_length = min(len(path_set[agent]) for path_set in path_sets)
        stop = common_length if include_goal else max(0, common_length - 1)
        points.extend((agent, timestep) for timestep in range(stop))
    return points


def run_adg_for_solution(
    param_meta,
    plan_name,
    info,
    delay_durations,
    delay_step_mode="all",
    injection_points=None,
    baseline_makespan=None,
):
    paths = info["paths"]
    path_dict = truncate_goal_waits(paths_to_agent_dict(paths))
    agents = sorted(path_dict.keys())
    planned_makespan = max(len(p) - 1 for p in path_dict.values())
    if baseline_makespan is None:
        baseline_makespan = planned_makespan

    makespan_group = f"M{baseline_makespan}"
    comparison_group = (
        f"{param_meta['grid']}_obs{param_meta['obs_pct']:02d}_"
        f"agents{param_meta['n_agents']:02d}_M{baseline_makespan}"
    )

    solution_file = csv_solution_path(info["solution_file"])
    rows = []

    if injection_points is None:
        injection_points = []
        for delayed_agent in agents:
            if delay_step_mode == "start":
                timesteps = [0]
            elif delay_step_mode in ("non-goal", "common-non-goal"):
                timesteps = list(range(max(0, len(path_dict[delayed_agent]) - 1)))
            else:
                timesteps = list(range(len(path_dict[delayed_agent])))
            injection_points.extend((delayed_agent, timestep) for timestep in timesteps)

    robustness_kappa = info.get("robustness_kappa", "")
    if robustness_kappa == "":
        applicable_durations = delay_durations
    elif robustness_kappa in delay_durations:
        applicable_durations = [robustness_kappa]
    else:
        applicable_durations = []

    for delay_duration in applicable_durations:
        for delayed_agent, delay_timestep in injection_points:
            if delayed_agent not in path_dict or delay_timestep >= len(path_dict[delayed_agent]):
                raise ValueError(
                    f"Injection ({delayed_agent}, {delay_timestep}) is not valid for {plan_name}"
                )
            if robustness_kappa == "":
                sim = simulate_adg_systematic(
                    paths,
                    delay_agent=delayed_agent,
                    delay_timestep=delay_timestep,
                    delay_duration=delay_duration,
                )
            else:
                sim = simulate_krobust_systematic(
                    paths,
                    delay_agent=delayed_agent,
                    delay_timestep=delay_timestep,
                    delay_duration=delay_duration,
                    robustness_kappa=robustness_kappa,
                )

            row = {
                "instance": param_meta["name"],
                "grid": param_meta["grid"],
                "obs_pct": param_meta["obs_pct"],
                "scenario": param_meta["scenario"],
                "n_agents": param_meta["n_agents"],
                "plan": plan_name,
                "makespan_group": makespan_group,
                "comparison_group": comparison_group,
                "baseline_makespan": baseline_makespan,
                "planned_makespan": sim["planned_makespan"],
                "delay_duration": delay_duration,
                "delay_step_mode": delay_step_mode,
                "robustness_kappa": robustness_kappa,
                "execution_policy": sim["execution_policy"],
                "delayed_agent": delayed_agent,
                "delay_timestep": delay_timestep,
                "injection_id": f"a{delayed_agent:03d}_t{delay_timestep:04d}",
                "executed_makespan": sim["executed_makespan"],
                "makespan_increase": sim["makespan_increase"],
                "makespan_increase_ratio": (
                    sim["makespan_increase"] / sim["planned_makespan"]
                    if sim["planned_makespan"] > 0
                    else 0.0
                ),
                "total_cascade": sim["total_cascade"],
                "total_cascade_all": sim["total_cascade_all"],
                "affected_agents": sim["affected_agents"],
                "n_adg_edges": sim["n_adg_edges"],
                "adg_has_cycle": sim["adg_has_cycle"],
                "converged": sim["converged"],
                "execution_safe": sim["execution_safe"],
                "conflict_type": sim["conflict_type"],
                "conflict_time": sim["conflict_time"],
                "solution_file": solution_file,
            }

            for a, value in sim["cascade_by_agent"].items():
                row[f"agent{a}_cascade"] = value

            rows.append(row)

    return rows


def aggregate_adg_details(detail_rows):
    groups = defaultdict(list)

    for row in detail_rows:
        key = (
            row["instance"],
            row["makespan_group"],
            row["plan"],
            row["delay_duration"],
            row.get("robustness_kappa", ""),
        )
        groups[key].append(row)

    summary = []

    for key, rows in groups.items():
        instance, makespan_group, plan, delay_duration, robustness_kappa = key
        first = rows[0]

        cascades = [r["total_cascade"] for r in rows]
        cascades_all = [r["total_cascade_all"] for r in rows]
        mk_increases = [r["makespan_increase"] for r in rows]
        mk_ratios = [r["makespan_increase_ratio"] for r in rows]
        affected = [r["affected_agents"] for r in rows]

        summary.append({
            "instance": instance,
            "makespan_group": makespan_group,
            "grid": first["grid"],
            "obs_pct": first["obs_pct"],
            "scenario": first["scenario"],
            "n_agents": first["n_agents"],
            "plan": plan,
            "delay_step_mode": first.get("delay_step_mode", ""),
            "delay_duration": delay_duration,
            "robustness_kappa": robustness_kappa,
            "baseline_makespan": first["baseline_makespan"],
            "planned_makespan": first["planned_makespan"],
            "n_trials": len(rows),
            "avg_cascade": round(mean(cascades), 6),
            "avg_normalized_cascade": round(
                mean(cascades) / first["baseline_makespan"], 12
            ),
            "median_cascade": round(median(cascades), 6),
            "max_cascade": max(cascades),
            "avg_cascade_all": round(mean(cascades_all), 6),
            "avg_makespan_increase": round(mean(mk_increases), 6),
            "median_makespan_increase": round(median(mk_increases), 6),
            "max_makespan_increase": max(mk_increases),
            "avg_makespan_increase_ratio": round(mean(mk_ratios), 6),
            "avg_affected_agents": round(mean(affected), 6),
        })

    summary.sort(
        key=lambda r: (
            natural_key(r["grid"]),
            int(r["obs_pct"]),
            int(r["scenario"]),
            int(r["n_agents"]),
            int(str(r["makespan_group"]).replace("M", "")),
            PLAN_ORDER.index(r["plan"]) if r["plan"] in PLAN_ORDER else 99,
            int(r["delay_duration"]),
        )
    )

    return summary


def aggregate_across_same_makespan(summary_rows):
    groups = defaultdict(list)

    for row in summary_rows:
        key = (
            row["makespan_group"],
            row["plan"],
            row["delay_duration"],
            row.get("robustness_kappa", ""),
        )
        groups[key].append(row)

    out = []

    for (makespan_group, plan, delay_duration, robustness_kappa), rows in groups.items():
        out.append({
            "makespan_group": makespan_group,
            "plan": plan,
            "delay_duration": delay_duration,
            "robustness_kappa": robustness_kappa,
            "n_instances": len(rows),
            "total_trials": sum(r["n_trials"] for r in rows),
            "avg_of_avg_cascade": round(mean(r["avg_cascade"] for r in rows), 6),
            "avg_normalized_cascade": round(
                mean(r["avg_normalized_cascade"] for r in rows), 12
            ),
            "avg_of_median_cascade": round(mean(r["median_cascade"] for r in rows), 6),
            "max_of_max_cascade": max(r["max_cascade"] for r in rows),
            "avg_of_avg_makespan_increase": round(
                mean(r["avg_makespan_increase"] for r in rows), 6
            ),
            "max_of_max_makespan_increase": max(r["max_makespan_increase"] for r in rows),
        })

    out.sort(
        key=lambda r: (
            int(str(r["makespan_group"]).replace("M", "")),
            PLAN_ORDER.index(r["plan"]) if r["plan"] in PLAN_ORDER else 99,
            int(r["delay_duration"]),
        )
    )

    return out


ADG_DETAILS_OMIT = {
    "delay_step_mode",
    "makespan_group",
    "comparison_group",
    "makespan_increase",
    "makespan_increase_ratio",
    "execution_policy",
    "robustness_kappa",
    "n_adg_edges",
    "adg_has_cycle",
    "converged",
    "execution_safe",
    "conflict_type",
    "conflict_time",
}


def save_csv(path: Path, rows, *, omit=()):
    ensure_dir(path.parent)

    if not rows:
        print(f"Nothing to save: {path}")
        return

    fields = []
    for row in rows:
        for key in row.keys():
            if key not in omit and key not in fields:
                fields.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="MAPF metric optimisation + systematic ADG delay experiments."
    )

    parser.add_argument(
        "--instances-dir",
        default="experiments",
        help="Directory generated by mapf_instances.py, containing params/.",
    )

    parser.add_argument(
        "--grid",
        default="5x5",
        help="Grid tag to process, e.g. 5x5. Use empty string for all.",
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Optional filter for number of agents.",
    )

    parser.add_argument(
        "--conjure-out-dir",
        default="outputs/conjure",
        help="Where Conjure solutions are stored.",
    )

    parser.add_argument(
        "--results-dir",
        default="outputs/results",
        help="Where CSV results are written.",
    )

    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip Conjure solving and parse existing solutions only.",
    )

    parser.add_argument(
        "--skip-adg",
        action="store_true",
        help="Only solve/parse solutions, do not run ADG delay experiment.",
    )

    parser.add_argument(
        "--solver",
        default=DEFAULT_SOLVER,
        help="Conjure solver backend.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout per Conjure solve in seconds.",
    )

    parser.add_argument(
        "--max-step-attempts",
        type=int,
        default=DEFAULT_MAX_STEP_ATTEMPTS,
        help="Max attempts while increasing makespan.",
    )

    parser.add_argument(
        "--max-metric4-attempts",
        type=int,
        default=DEFAULT_MAX_METRIC4_ATTEMPTS,
        help="Max incumbent-improvement attempts for metric4.",
    )

    parser.add_argument(
        "--max-robust-attempts",
        type=int,
        default=DEFAULT_MAX_STEP_ATTEMPTS,
        help="Maximum horizons tested while finding each kappa-robust plan.",
    )

    parser.add_argument(
        "--delay-durations",
        "--k-values",
        dest="delay_durations",
        default="5,10,15",
        help="Injected unavoidable-delay durations delta, e.g. 5,10,15.",
    )

    parser.add_argument(
        "--robustness-values",
        default="none",
        help=(
            "Comma-separated robustness guarantees kappa. "
            "Use 'match' for the delay durations or 'none' (default) to omit "
            "the optional kappa-robust comparison."
        ),
    )

    parser.add_argument(
        "--skip-krobust",
        action="store_true",
        help="Deprecated alias for --robustness-values none.",
    )

    parser.add_argument(
        "--delay-step-mode",
        choices=["common-all", "common-non-goal", "all", "non-goal", "start"],
        default="all",
        help=(
            "Delay timestep sweep. "
            "'all' (literal PDF protocol) = every timestep in each plan; "
            "'common-all' = paired agent-time indices valid in every plan; "
            "'common-non-goal' = paired agent-time indices excluding goal arrival; "
            "'non-goal' = plan-specific states excluding goal arrival; "
            "'start' = only timestep 0."
        ),
    )

    args = parser.parse_args()

    work_dir = Path(__file__).parent.resolve()
    repo_dir = work_dir.parent if work_dir.name == "src" else work_dir
    os.chdir(repo_dir)

    instances_dir = Path(args.instances_dir).resolve()
    conjure_out_dir = Path(args.conjure_out_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    ensure_dir(conjure_out_dir)
    ensure_dir(results_dir)

    grid_filter = args.grid if args.grid else None
    try:
        delay_durations = [
            int(x.strip()) for x in args.delay_durations.split(",") if x.strip()
        ]
    except ValueError:
        parser.error("--delay-durations must be a comma-separated list of integers")
    if not delay_durations:
        parser.error("--delay-durations must contain at least one integer")
    if any(value < 0 for value in delay_durations):
        parser.error("--delay-durations must contain non-negative integers")

    robustness_text = args.robustness_values.strip().lower()
    if args.skip_krobust or robustness_text in ("", "none"):
        robustness_values = []
    elif robustness_text == "match":
        robustness_values = list(delay_durations)
    else:
        try:
            robustness_values = [
                int(x.strip())
                for x in args.robustness_values.split(",")
                if x.strip()
            ]
        except ValueError:
            parser.error(
                "--robustness-values must be 'none', 'match', or a "
                "comma-separated list of integers"
            )
    if any(value < 0 for value in robustness_values):
        parser.error("--robustness-values must contain non-negative integers")

    try:
        model_paths = validate_required_models(
            work_dir, robustness_values
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    started_at = datetime.now(timezone.utc).isoformat()
    metadata_path = results_dir / "run_metadata.json"
    run_metadata = {
        "schema_version": 1,
        "status": "started",
        "started_at_utc": started_at,
        "python_version": platform.python_version(),
        "conjure_version": command_version("conjure"),
        "solver": args.solver,
        "instances_dir": repository_relative_path(instances_dir),
        "conjure_out_dir": repository_relative_path(conjure_out_dir),
        "results_dir": repository_relative_path(results_dir),
        "grid_filter": grid_filter,
        "agent_filter": args.agents,
        "delay_durations": delay_durations,
        "delay_step_mode": args.delay_step_mode,
        "robustness_values": robustness_values,
        "parse_only": args.parse_only,
        "skip_adg": args.skip_adg,
        "timeout_seconds": args.timeout,
        "max_step_attempts": args.max_step_attempts,
        "max_metric4_attempts": args.max_metric4_attempts,
        "max_robust_attempts": args.max_robust_attempts,
        "models": {
            path.name: sha256_file(path)
            for path in model_paths
        },
    }
    write_run_metadata(metadata_path, run_metadata)

    print("============================================================")
    print("MAPF full experiment with systematic ADG delays")
    print("============================================================")
    print(f"Work dir:       {work_dir}")
    print(f"Instances dir:  {instances_dir}")
    print(f"Conjure output: {conjure_out_dir}")
    print(f"Results dir:    {results_dir}")
    print(f"Grid filter:    {grid_filter}")
    print(f"Agent filter:   {args.agents}")
    print(f"Delay delta:    {delay_durations}")
    print(f"Robust kappa:   {robustness_values or 'skipped'}")
    print(f"Delay mode:     {args.delay_step_mode}")
    print("")

    if not args.parse_only and not command_exists("conjure"):
        print("ERROR: 'conjure' command not found.")
        print("Run this script inside WSL/Linux where Conjure is installed, for example:")
        print("  From the repository root:")
        print("  python3 src/experiment.py --instances-dir experiments --grid 5x5")
        run_metadata.update({
            "status": "failed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": "conjure command not found",
        })
        write_run_metadata(metadata_path, run_metadata)
        sys.exit(2)

    params = collect_params(instances_dir, grid_filter=grid_filter, agents_filter=args.agents)
    run_metadata["instance_count"] = len(params)
    write_run_metadata(metadata_path, run_metadata)

    print("=== STEP 1: COLLECTING .PARAM FILES ===")
    print(f"  {len(params)} param files ready")

    if not params:
        print("ERROR: no parameters found.")
        run_metadata.update({
            "status": "failed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": "no parameter files found",
        })
        write_run_metadata(metadata_path, run_metadata)
        sys.exit(2)

    all_solutions = {}
    solver_rows = []

    if not args.parse_only:
        print("\n=== STEP 2: RUNNING CONJURE SOLVER ===")

        for idx, pm in enumerate(params, start=1):
            print(f"\n[{idx}/{len(params)}] {pm['name']}")

            sol_info = run_solver_for_instance(
                pm,
                work_dir=work_dir,
                conjure_out_dir=conjure_out_dir,
                solver=args.solver,
                timeout=args.timeout,
                max_step_attempts=args.max_step_attempts,
                max_metric4_attempts=args.max_metric4_attempts,
                robustness_values=robustness_values,
                max_robust_attempts=args.max_robust_attempts,
            )

            all_solutions[pm["name"]] = sol_info
            save_solution_manifest(pm, sol_info, conjure_out_dir)

            for plan_name, info in sol_info.items():
                display_name = info.get("plan_label", plan_name)
                solver_rows.append(make_solution_summary_row(pm, display_name, info))

    else:
        print("\n=== STEP 2: PARSING EXISTING SOLUTIONS ===")

        for idx, pm in enumerate(params, start=1):
            sol_info = load_existing_solutions_for_instance(
                pm,
                conjure_out_dir,
                work_dir,
                robustness_values=robustness_values,
            )
            all_solutions[pm["name"]] = sol_info

            print(f"[{idx}/{len(params)}] {pm['name']}: loaded {len(sol_info)} solution types")

            for plan_name, info in sol_info.items():
                display_name = info.get("plan_label", plan_name)
                solver_rows.append(make_solution_summary_row(pm, display_name, info))

    save_csv(results_dir / "solver_solution_summary.csv", solver_rows)

    expected_solution_rows = len(params) * (len(PLAN_ORDER) + len(robustness_values))
    solutions_complete = len(solver_rows) == expected_solution_rows
    if not solutions_complete:
        print(
            f"WARNING: expected {expected_solution_rows} solution summary rows "
            f"({len(params)} instances x "
            f"{len(PLAN_ORDER) + len(robustness_values)} plan configurations), "
            f"but got {len(solver_rows)}"
        )
        print("         Check warnings above and solution manifests for missing plans.")
    else:
        print(f"Solution summary complete: {len(solver_rows)}/{expected_solution_rows} rows")

    if args.skip_adg:
        print("\nSkipping ADG experiment.")
        run_metadata.update({
            "status": "complete" if solutions_complete else "incomplete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "solution_summary_rows": len(solver_rows),
            "expected_solution_summary_rows": expected_solution_rows,
            "adg_detail_rows": 0,
        })
        write_run_metadata(metadata_path, run_metadata)
        if not solutions_complete:
            sys.exit(1)
        return

    print("\n=== STEP 3: SYSTEMATIC ADG DELAY EXPERIMENT ===")

    detail_rows = []

    for idx, pm in enumerate(params, start=1):
        sol_info = all_solutions.get(pm["name"], {})

        if not sol_info:
            print(f"[{idx}/{len(params)}] {pm['name']}: SKIP no solutions")
            continue

        print(f"[{idx}/{len(params)}] {pm['name']}: ADG on {len(sol_info)} plans")

        paired_points = None
        if args.delay_step_mode.startswith("common-"):
            available_infos = list(sol_info.values())
            paired_points = build_paired_injection_points(
                available_infos,
                include_goal=args.delay_step_mode == "common-all",
            )
            print(f"  paired delay injection points: {len(paired_points)}")

        baseline_paths = truncate_goal_waits(
            paths_to_agent_dict(sol_info["makespan"]["paths"])
        )
        baseline_makespan = max(len(path) - 1 for path in baseline_paths.values())

        ordered_keys = [name for name in PLAN_ORDER if name in sol_info]
        ordered_keys.extend(
            sorted(name for name in sol_info if name not in PLAN_ORDER)
        )
        for internal_name in ordered_keys:
            info = sol_info[internal_name]
            plan_name = info.get("plan_label", internal_name)
            rows = run_adg_for_solution(
                pm,
                plan_name,
                info,
                delay_durations=delay_durations,
                delay_step_mode=args.delay_step_mode,
                injection_points=paired_points,
                baseline_makespan=baseline_makespan,
            )
            detail_rows.extend(rows)

    summary_rows = aggregate_adg_details(detail_rows)
    makespan_only_rows = aggregate_across_same_makespan(summary_rows)
    invalid_adg_rows = [
        row
        for row in detail_rows
        if not row["converged"] or not row["execution_safe"] or row["adg_has_cycle"]
    ]
    adg_complete = bool(detail_rows) and not invalid_adg_rows

    print("\n=== STEP 4: SAVING RESULTS ===")
    save_csv(
        results_dir / "adg_delay_details.csv",
        detail_rows,
        omit=ADG_DETAILS_OMIT,
    )
    save_csv(results_dir / "adg_delay_summary_by_makespan.csv", summary_rows)
    save_csv(results_dir / "adg_delay_same_makespan_presentation.csv", makespan_only_rows)

    print("\nDone.")
    print(f"Solution summary rows: {len(solver_rows)}")
    print(f"ADG detail rows:       {len(detail_rows)}")
    print(f"ADG summary rows:      {len(summary_rows)}")
    print(f"Presentation rows:     {len(makespan_only_rows)}")
    if invalid_adg_rows:
        print(
            f"ERROR: {len(invalid_adg_rows)} ADG rows were unsafe, cyclic, "
            "or did not converge."
        )
    run_complete = solutions_complete and adg_complete
    run_metadata.update({
        "status": "complete" if run_complete else "incomplete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "solution_summary_rows": len(solver_rows),
        "expected_solution_summary_rows": expected_solution_rows,
        "adg_detail_rows": len(detail_rows),
        "adg_summary_rows": len(summary_rows),
        "presentation_rows": len(makespan_only_rows),
        "invalid_adg_rows": len(invalid_adg_rows),
    })
    write_run_metadata(metadata_path, run_metadata)
    if not run_complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
