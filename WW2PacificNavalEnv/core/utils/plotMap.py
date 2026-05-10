import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch


def plot_map(json_path: str, figsize=(14, 9), show_grid: bool = True, save_path: str | None = None):
    """Pretty map plot: blue=water, green=land, grey=ports.
    - Origin (0,0) at bottom-left
    - Major ticks/labels every 10 tiles
    - Very thin grid lines every 1 tile (no labels)"""

    with open(json_path) as f:
        data = json.load(f)

    w, h = data["width"], data["height"]
    grid = np.zeros((h, w), dtype=int)  # 0=water, 1=land, 2=port

    for x, y in data["land"]:
        grid[y, x] = 1

    for player_ports in data["ports"].values():
        for x, y in player_ports:
            grid[y, x] = 2

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.colors.ListedColormap(["#1e88e5", "#43a047", "#616161"])
    # extent makes each cell span exactly 1 unit with boundaries on integers,
    # so tick labels and grid lines align with cell edges (corner of map = (0,0)).
    ax.imshow(grid, cmap=cmap, interpolation="nearest", origin="lower", extent=[0, w, 0, h])

    # Hard bounds
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)

    if show_grid:
        # Major ticks (labeled every 10)
        ax.set_xticks(np.arange(0, w + 1, 10))
        ax.set_yticks(np.arange(0, h + 1, 10))

        # Minor ticks for fine grid (no labels) — fixed to align perfectly with major
        ax.set_xticks(np.arange(0, w + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, h + 1, 1), minor=True)

        # Very thin fine grid
        ax.grid(True, which="minor", color="white", linestyle="-", linewidth=0.25, alpha=0.5)
        # Slightly stronger major grid
        ax.grid(True, which="major", color="white", linestyle="-", linewidth=0.6, alpha=0.7)

    ax.set_title(
        f'Map (seed={data.get("seed", "unknown")}) | '
        f'{len(data["land"])} land | '
        f'{sum(len(p) for p in data["ports"].values())} ports',
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Legend
    legend_elements = [
        Patch(facecolor="#1e88e5", label="Water"),
        Patch(facecolor="#43a047", label="Land"),
        Patch(facecolor="#616161", label="Port (Base)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")

    plt.show()
