"""Generate maze visualizations for SRE incident scenarios.

Approach: generate a real maze, solve it, place doors along the solution
path (green) and on wrong-turn branches (red/gray).

- IN = entrance on the left wall
- Star = EXIT on the right wall (only one)
- Green doors = correct actions, placed IN corridors along solution path
- Red doors + monster face = trap actions, placed IN branch corridors
- Gray doors = no_effect actions, placed IN branch corridors
- Doors are same size as corridor cells (single cell)
- Paths between doors are arbitrarily long
"""

import json
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from collections import deque


def generate_maze(width, height, seed=42):
    """Recursive backtracker. Returns grid (True=wall, False=passage).
    Grid is (2*height+1) x (2*width+1). Passages at odd coords."""
    random.seed(seed)
    w, h = width, height
    gw = w * 2 + 1
    gh = h * 2 + 1
    grid = np.ones((gh, gw), dtype=bool)

    def carve(cx, cy):
        grid[cy * 2 + 1, cx * 2 + 1] = False
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny * 2 + 1, nx * 2 + 1]:
                grid[cy * 2 + 1 + dy, cx * 2 + 1 + dx] = False
                carve(nx, ny)

    carve(0, 0)
    return grid


def bfs_path(grid, start, end):
    """Shortest path through passages. Returns list of (x, y) or []."""
    gh, gw = grid.shape
    queue = deque([start])
    prev = {start: None}
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == end:
            path = []
            pos = end
            while pos is not None:
                path.append(pos)
                pos = prev[pos]
            return path[::-1]
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < gw and 0 <= ny < gh and not grid[ny, nx] and (nx, ny) not in prev:
                prev[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))
    return []


def find_branches(grid, solution_set):
    """Find cells that branch OFF the solution path into side corridors.
    Returns list of (branch_cell, parent_on_solution)."""
    gh, gw = grid.shape
    branches = []
    for sx, sy in solution_set:
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = sx + dx, sy + dy
            if (nx, ny) not in solution_set and 0 <= nx < gw and 0 <= ny < gh and not grid[ny, nx]:
                branches.append(((nx, ny), (sx, sy)))
    return branches


def count_doors(states):
    """Count unique edges by outcome type from the state graph."""
    green = red = gray = 0
    seen = set()
    for sn, sd in states.items():
        if sd.get('is_resolved'):
            continue
        for a in sd.get('actions', []):
            edge = (sn, a['next_state'], a['outcome'])
            key = (sn, a['next_state'])
            if key in seen:
                continue
            seen.add(key)
            if a['outcome'] in ('progress', 'recovery'):
                green += 1
            elif a['outcome'] == 'worsened':
                red += 1
            else:
                gray += 1
    return green, red, gray


def draw_scenario(scenario, out_dir):
    sid = scenario['id']
    short = sid.replace('_001', '').replace('_h002', '')
    diff = scenario['difficulty']
    rem = scenario['failure']['remediation']
    states = rem['states']
    opt_steps = rem.get('optimal_steps', 0)

    green_count, red_count, gray_count = count_doors(states)
    total_doors = green_count + red_count + gray_count

    # Maze size: smaller = bigger cells = doors clearly visible
    maze_w = max(8, green_count * 3 + red_count + 4)
    maze_h = max(6, total_doors + 2)
    grid = generate_maze(maze_w, maze_h, seed=hash(sid) % 99999)
    gh, gw = grid.shape

    # --- Open entrance (left wall) and exit (right wall) ---
    # Find a passage cell in column 1, open wall at column 0
    entrance = None
    for y in range(1, gh - 1):
        if not grid[y, 1]:
            grid[y, 0] = False
            entrance = (0, y)
            break
    if not entrance:
        grid[1, 0] = False
        grid[1, 1] = False
        entrance = (0, 1)

    # Find a passage cell in second-to-last column, open wall at last column
    exit_pos = None
    for y in range(1, gh - 1):
        if not grid[y, gw - 2]:
            grid[y, gw - 1] = False
            exit_pos = (gw - 1, y)
            break
    if not exit_pos:
        grid[gh - 2, gw - 1] = False
        grid[gh - 2, gw - 2] = False
        exit_pos = (gw - 1, gh - 2)

    # --- Solve maze ---
    solution = bfs_path(grid, entrance, exit_pos)
    if not solution or len(solution) < 3:
        print(f"  WARNING: no solution for {short}, skipping")
        return

    solution_set = set(solution)

    # --- Place green doors along solution path ---
    doors = []  # (x, y, type)
    if green_count > 0:
        # Evenly space green doors along solution, avoiding first/last cells
        usable = solution[2:-2]  # skip entrance/exit cells
        if len(usable) >= green_count:
            step = len(usable) / (green_count + 1)
            for i in range(1, green_count + 1):
                idx = int(i * step)
                idx = min(idx, len(usable) - 1)
                dx, dy = usable[idx]
                doors.append((dx, dy, 'progress'))
        else:
            # Path too short — place what we can
            for i, (dx, dy) in enumerate(usable):
                if i < green_count:
                    doors.append((dx, dy, 'progress'))

    # --- Find branches off solution for red/gray doors ---
    branches = find_branches(grid, solution_set)
    random.seed(hash(sid) + 1)
    random.shuffle(branches)

    # Deduplicate: one door per branch cell
    used_cells = set(d[:2] for d in doors)
    placed_red = 0
    placed_gray = 0
    for (bx, by), _ in branches:
        if (bx, by) in used_cells:
            continue
        if placed_red < red_count:
            doors.append((bx, by, 'worsened'))
            used_cells.add((bx, by))
            placed_red += 1
        elif placed_gray < gray_count:
            doors.append((bx, by, 'no_effect'))
            used_cells.add((bx, by))
            placed_gray += 1
        if placed_red >= red_count and placed_gray >= gray_count:
            break

    # === DRAW ===
    cell = 0.5
    fig_w = gw * cell + 0.5
    fig_h = gh * cell + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#0d1117')

    # Draw maze cells — dark passages, light walls (classic maze style)
    for y in range(gh):
        for x in range(gw):
            sy = gh - y - 1
            if grid[y, x]:
                # Wall — light/cream
                c = '#c8c8c8'
                ec = '#b0b0b0'
            else:
                # Passage — dark
                c = '#1a1a2e'
                ec = '#16213e'
            ax.add_patch(patches.Rectangle(
                (x, sy), 1, 1, facecolor=c, edgecolor=ec, linewidth=0.15))

    # Draw doors (same size as corridor cells — single cell)
    for dx, dy, outcome in doors:
        sy = gh - dy - 1
        if outcome in ('progress', 'recovery'):
            # Green door
            ax.add_patch(patches.Rectangle(
                (dx, sy), 1, 1, facecolor='#00b894', edgecolor='#00cec9',
                linewidth=1.5, zorder=3))
        elif outcome == 'worsened':
            # Red door with monster
            ax.add_patch(patches.Rectangle(
                (dx, sy), 1, 1, facecolor='#d63031', edgecolor='#e17055',
                linewidth=1.5, zorder=3))
            ax.text(dx + 0.5, sy + 0.5, '>.<',
                    fontsize=max(4, cell * 14), ha='center', va='center',
                    color='#fdcb6e', fontweight='bold', family='monospace',
                    zorder=4)
        else:
            # Gray door (no_effect)
            ax.add_patch(patches.Rectangle(
                (dx, sy), 1, 1, facecolor='#636e72', edgecolor='#b2bec3',
                linewidth=1, zorder=3))

    # IN marker at entrance
    ix, iy = entrance
    sy_in = gh - iy - 1
    ax.text(ix + 0.5, sy_in + 0.5, 'IN', fontsize=7,
            ha='center', va='center', color='white',
            fontweight='bold', family='monospace', zorder=5,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#d63031',
                      edgecolor='#e17055', alpha=0.95))

    # Star + EXIT at exit (right wall)
    ex, ey = exit_pos
    sy_ex = gh - ey - 1
    ax.text(ex + 0.5, sy_ex + 0.5, '\u2605', fontsize=12,
            ha='center', va='center', color='#ffeaa7',
            fontweight='bold', zorder=5)
    ax.text(ex + 0.5, sy_ex - 0.6, 'EXIT', fontsize=5,
            ha='center', va='center', color='#fdcb6e',
            fontweight='bold', family='monospace', zorder=5)

    ax.set_xlim(-0.5, gw + 0.5)
    ax.set_ylim(-1, gh + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    tc = {'easy': '#00b894', 'medium': '#fdcb6e', 'hard': '#d63031', 'expert': '#6c5ce7'}[diff]
    ax.set_title(
        f"{short}\n{diff.upper()} | {opt_steps}-step | "
        f"{green_count}\u2714 {red_count}\u2718 {gray_count}\u25cb doors",
        color=tc, fontsize=9, fontweight='bold', pad=8, family='monospace')

    plt.tight_layout()
    plt.savefig(out_dir / f"{short}.png", dpi=250, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


if __name__ == '__main__':
    out_dir = Path("outputs/maze_v7")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open('scenarios/incidents_v2.jsonl') as f:
        for line in f:
            sc = json.loads(line)
            short = sc['id'].replace('_001', '').replace('_h002', '')
            draw_scenario(sc, out_dir)
            print(f"  {short}.png")

    print(f"\nDone! {out_dir}")
