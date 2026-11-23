#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path


DEFAULT_SEED = 73
DEFAULT_OUTPUT_DIR = Path("outputs/generated_instances")
GRID_SIZES = [(5, 5)]
OBSTACLE_PERCENTAGES = [30, 25, 20, 15, 10, 5, 0]
DEFAULT_SCENARIOS = 3

AGENT_COUNTS = {(5, 5): [2, 3, 4],}


def is_free(grid: list[list[int]], row: int, col: int) -> bool:
    height, width = len(grid), len(grid[0])
    return 0 <= row < height and 0 <= col < width and grid[row][col] == 0


def reachable_cells(grid: list[list[int]], start: tuple[int, int]) -> set[tuple[int, int]]:
    visited = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + d_row, col + d_col)
            if is_free(grid, *neighbor) and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def is_connected(grid: list[list[int]]) -> bool:
    height, width = len(grid), len(grid[0])
    first_free = next(
        (
            (row, col)
            for row in range(height)
            for col in range(width)
            if grid[row][col] == 0
        ),
        None,
    )
    if first_free is None:
        return False
    free_count = sum(cell == 0 for row in grid for cell in row)
    return len(reachable_cells(grid, first_free)) == free_count


def obstacle_count(height: int, width: int, nominal_percentage: int) -> int:
    return height * width * nominal_percentage // 100

def generate_base_map(height: int,width: int,nominal_percentage: int,rng: random.Random,max_tries: int = 500,) -> list[list[int]]:
    n_obstacles = obstacle_count(height, width, nominal_percentage)
    cells = [(row, col) for row in range(height) for col in range(width)]

    for _ in range(max_tries):
        grid = [[0] * width for _ in range(height)]
        for row, col in rng.sample(cells, n_obstacles):
            grid[row][col] = 1
        if is_connected(grid):
            return grid

    raise RuntimeError(
        f"Could not generate a connected {height}x{width} map at nominal "
        f"density {nominal_percentage}% after {max_tries} attempts."
    )


def derive_density_variant(base_grid: list[list[int]],nominal_percentage: int,rng: random.Random,) -> list[list[int]]:
    
    height, width = len(base_grid), len(base_grid[0])
    grid = [row[:] for row in base_grid]
    current_obstacles = [
        (row, col)
        for row in range(height)
        for col in range(width)
        if grid[row][col] == 1
    ]
    target_count = obstacle_count(height, width, nominal_percentage)

    if len(current_obstacles) < target_count:
        raise ValueError("A density variant cannot add obstacles to the base map.")

    rng.shuffle(current_obstacles)
    for row, col in current_obstacles[: len(current_obstacles) - target_count]:
        grid[row][col] = 0

    if not is_connected(grid):
        raise AssertionError("Removing obstacles unexpectedly disconnected free space.")
    return grid


def generate_scenario(grid: list[list[int]],n_agents: int,rng: random.Random,) -> list[tuple[int, int, int, int]]:
    free_cells = [
        (row, col)
        for row in range(len(grid))
        for col in range(len(grid[0]))
        if grid[row][col] == 0
    ]
    if len(free_cells) < 2 * n_agents:
        raise ValueError(
            f"Need {2 * n_agents} distinct free cells for {n_agents} agents, "
            f"but only {len(free_cells)} are available."
        )

    positions = rng.sample(free_cells, 2 * n_agents)
    starts = positions[:n_agents]
    goals = positions[n_agents:]
    return [
        (start_row, start_col, goal_row, goal_col)
        for (start_row, start_col), (goal_row, goal_col) in zip(starts, goals)
    ]


def shortest_path_length(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> int:
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (row, col), distance = queue.popleft()
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + d_row, col + d_col)
            if not is_free(grid, *neighbor) or neighbor in visited:
                continue
            if neighbor == goal:
                return distance + 1
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    raise ValueError(f"No path exists from {start} to {goal}.")


def write_map_file(path: Path, grid: list[list[int]]) -> None:
    height, width = len(grid), len(grid[0])
    lines = ["type octile", f"height {height}", f"width {width}", "map"]
    lines.extend("".join("T" if cell else "." for cell in row) for row in grid)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scenario_file(path: Path,grid: list[list[int]],agents: list[tuple[int, int, int, int]],grid_name: str,) -> None:
    height, width = len(grid), len(grid[0])
    lines = ["version 1"]
    for bucket, (start_row, start_col, goal_row, goal_col) in enumerate(agents):
        lines.append(
            f"{bucket}\t{grid_name}\t{width}\t{height}\t{start_col}\t"
            f"{start_row}\t{goal_col}\t{goal_row}\t0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conjure_parameter(path: Path,grid: list[list[int]],agents: list[tuple[int, int, int, int]],extra_states: int = 2,) -> None:
    height, width = len(grid), len(grid[0])
    lower_bound_moves = max(
        shortest_path_length(grid, (sr, sc), (gr, gc))
        for sr, sc, gr, gc in agents
    )
    initial_states = lower_bound_moves + extra_states

    map_rows = ["[" + ", ".join(str(value) for value in row) + "]" for row in grid]
    map_text = "[\n  " + ",\n  ".join(map_rows) + "\n]"
    agent_rows = [
        "  record { "
        f"start_x = {sr + 1}, start_y = {sc + 1}, "
        f"goal_x = {gr + 1}, goal_y = {gc + 1} "
        "}"
        for sr, sc, gr, gc in agents
    ]
    parameter_text = (
        f"letting height be {height}\n"
        f"letting width be {width}\n"
        f"letting map be\n{map_text}\n\n"
        "letting agents be\n[\n"
        + ",\n".join(agent_rows)
        + f"\n]\n\nletting steps be {initial_states}\n"
    )
    path.write_text(parameter_text, encoding="utf-8")


def build_instances(output_dir: Path, seed: int, scenarios: int) -> list[dict]:
    rng = random.Random(seed)
    maps_dir = output_dir / "maps"
    scenarios_dir = output_dir / "scenarios"
    parameters_dir = output_dir / "params"
    for directory in (maps_dir, scenarios_dir, parameters_dir):
        directory.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    realized_densities: dict[str, dict] = {}

    for height, width in GRID_SIZES:
        size_tag = f"{height}x{width}"
        counts = AGENT_COUNTS[(height, width)]
        max_agents = max(counts)
        base_density = max(OBSTACLE_PERCENTAGES)

        base_grid = generate_base_map(height, width, base_density, rng)

        variants = {
            density: (
                [row[:] for row in base_grid]
                if density == base_density
                else derive_density_variant(base_grid, density, rng)
            )
            for density in OBSTACLE_PERCENTAGES
        }

        for density in OBSTACLE_PERCENTAGES:
            grid = variants[density]
            n_obstacles = sum(cell == 1 for row in grid for cell in row)
            actual_percentage = 100.0 * n_obstacles / (height * width)
            realized_densities[f"{size_tag}:{density}"] = {
                "nominal_percentage": density,
                "obstacle_count": n_obstacles,
                "actual_percentage": actual_percentage,
            }

            variant_tag = f"{size_tag}_{density:02d}pct"
            map_path = maps_dir / size_tag / f"map_{density:02d}pct.map"
            map_path.parent.mkdir(parents=True, exist_ok=True)
            write_map_file(map_path, grid)

            for scenario_index in range(1, scenarios + 1):
                pairs = generate_scenario(grid, max_agents, rng)

                scenario_name = f"{variant_tag}_scen{scenario_index:02d}"
                scenario_path = scenarios_dir / size_tag / f"{scenario_name}.scen"
                scenario_path.parent.mkdir(parents=True, exist_ok=True)
                write_scenario_file(scenario_path, grid, pairs, map_path.name)

                for n_agents in counts:
                    parameter_name = f"{scenario_name}_agents{n_agents:02d}"
                    parameter_path = (
                        parameters_dir
                        / size_tag
                        / variant_tag
                        / f"{parameter_name}.param"
                    )
                    parameter_path.parent.mkdir(parents=True, exist_ok=True)
                    write_conjure_parameter(parameter_path, grid, pairs[:n_agents])

                    index_rows.append(
                        {
                            "grid": size_tag,
                            "nominal_obs_pct": density,
                            "obstacle_count": n_obstacles,
                            "actual_obs_pct": actual_percentage,
                            "scenario": scenario_index,
                            "n_agents": n_agents,
                            "seed": seed,
                            "map_file": str(map_path.relative_to(output_dir)),
                            "scen_file": str(scenario_path.relative_to(output_dir)),
                            "param_file": str(parameter_path.relative_to(output_dir)),
                        }
                    )

    if not index_rows:
        raise RuntimeError("No MAPF configurations were generated.")

    index_path = output_dir / "instance_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    config = {
        "seed": seed,
        "grid_sizes": [list(size) for size in GRID_SIZES],
        "nominal_obstacle_percentages": OBSTACLE_PERCENTAGES,
        "scenarios_per_map": scenarios,
        "agent_counts": {
            f"{height}x{width}": AGENT_COUNTS[(height, width)]
            for height, width in GRID_SIZES
        },
        "variant_design": (
            "Each density variant is independently derived by removing obstacles "
            "from the same connected maximum-density base map."
        ),
        "realized_densities": realized_densities,
    }
    (output_dir / "generation_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return index_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the reproducible 5x5 MAPF configurations."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    args = parser.parse_args(argv)

    if args.scenarios < 1:
        parser.error("--scenarios must be at least 1")

    rows = build_instances(args.output_dir, args.seed, args.scenarios)
    print(f"Generated {len(rows)} configurations in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
