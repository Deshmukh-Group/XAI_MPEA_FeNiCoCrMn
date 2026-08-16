#!/usr/bin/env python3
"""Generate the two publication figures from the accompanying 15-row CSV."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress, pearsonr


HERE = Path(__file__).resolve().parent
DATA = HERE / "pair_plot_data.csv"
ELEMENTS = ["Mn", "Cr", "Co", "Fe", "Ni"]


def readable_text(rgba):
    r, g, b = rgba[:3]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else "black"


def load_data():
    data = pd.read_csv(DATA)
    if len(data) != 15 or not data["Pair_Type"].is_unique:
        raise ValueError("pair_plot_data.csv must contain 15 unique pair types")

    energy = np.zeros((5, 5))
    energy_sd = np.zeros((5, 5))
    correlation = np.zeros((5, 5))
    for row in data.itertuples(index=False):
        i, j = (int(value) - 1 for value in row.Pair_Type.split("-"))
        for a, b in ((i, j), (j, i)):
            energy[a, b] = row.Mean_Energy_eV
            energy_sd[a, b] = row.SD_Energy_eV
            correlation[a, b] = row.Pair_Count_Bulk_Pearson_r
    return data, energy, energy_sd, correlation


def plot_scatter(data):
    x = data["Mean_Energy_eV"].to_numpy()
    y = data["Pair_Count_Bulk_Pearson_r"].to_numpy()
    r, p = pearsonr(x, y)
    fit = linregress(x, y)
    x_fit = np.linspace(x.min() - 0.02, x.max() + 0.02, 200)

    fig, ax = plt.subplots(figsize=(9.2, 9.2))
    ax.set_box_aspect(1)
    points = ax.scatter(
        x, y, c=y, cmap="RdBu_r", vmin=-1, vmax=1, s=135,
        edgecolor="black", linewidth=0.8, zorder=3,
    )
    ax.plot(
        x_fit, fit.intercept + fit.slope * x_fit, "--", color="0.30",
        linewidth=1.6, zorder=2, label="Linear trend across the 15 pair types",
    )
    ax.axhline(0, color="0.55", linewidth=1, zorder=1)

    offsets = {
        "Mn-Mn": (-61, 10), "Mn-Cr": (9, -20), "Mn-Co": (8, 8),
        "Mn-Fe": (8, 8), "Mn-Ni": (9, -20), "Cr-Cr": (8, 8),
        "Cr-Co": (10, 11), "Cr-Fe": (10, -20), "Cr-Ni": (10, -22),
        "Co-Co": (10, -2), "Co-Fe": (8, 8), "Co-Ni": (10, 10),
        "Fe-Fe": (-57, -22), "Fe-Ni": (8, -17), "Ni-Ni": (10, -17),
    }
    for row in data.itertuples(index=False):
        ax.annotate(
            row.Element_Pair, (row.Mean_Energy_eV, row.Pair_Count_Bulk_Pearson_r),
            xytext=offsets[row.Element_Pair], textcoords="offset points",
            fontsize=10.5, fontweight="bold", zorder=4,
        )

    ax.text(
        0.025, 0.035, f"Across-pair Pearson r = {r:.2f}\np = {p:.3g}",
        transform=ax.transAxes, fontsize=11.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.55", alpha=0.92),
    )
    ax.set(
        xlabel="Average Pair Interaction Energy (eV)",
        ylabel="Pearson Correlation: Pair Count vs Bulk Modulus",
        xlim=(x.min() - 0.055, x.max() + 0.055), ylim=(-1, 1),
    )
    ax.xaxis.label.set(fontsize=16, fontweight="bold")
    ax.yaxis.label.set(fontsize=16, fontweight="bold")
    ax.xaxis.labelpad = ax.yaxis.labelpad = 10
    ax.grid(color="0.88", linewidth=0.8, zorder=0)
    ax.tick_params(labelsize=11.5)
    ax.legend(loc="upper right", fontsize=10.5, frameon=True)
    cax = ax.inset_axes([1.035, 0, 0.055, 1], transform=ax.transAxes)
    bar = fig.colorbar(points, cax=cax)
    bar.set_label("Pearson Correlation", fontsize=14, fontweight="bold", labelpad=10)
    bar.ax.tick_params(labelsize=11.5)
    fig.tight_layout()
    fig.savefig(HERE / "average_energy_vs_pair_count_correlation.png", dpi=800, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(energy, energy_sd, correlation):
    energy_cmap = plt.colormaps["viridis_r"]
    energy_norm = colors.Normalize(energy.min(), energy.max())
    corr_cmap = plt.colormaps["RdBu_r"]
    corr_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 14})
    fig, ax = plt.subplots(figsize=(12, 12))

    for i in range(5):
        for j in range(5):
            if i > j:
                value, sd = energy[i, j], energy_sd[i, j]
                face = energy_cmap(energy_norm(value))
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=1.5))
                ax.text(j + .5, i + .5, f"{value:.3f}\n± {sd:.3f}", ha="center", va="center",
                        fontsize=14, color=readable_text(face), fontweight="bold")
            elif i < j:
                value = correlation[i, j]
                face = corr_cmap(corr_norm(value))
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=1.5))
                ax.text(j + .5, i + .5, f"{value:.2f}", ha="center", va="center",
                        fontsize=15, color=readable_text(face), fontweight="bold")
            else:
                eface = energy_cmap(energy_norm(energy[i, i]))
                cface = corr_cmap(corr_norm(correlation[i, i]))
                ax.add_patch(Polygon([(j, i + 1), (j + 1, i + 1), (j, i)], facecolor=eface,
                                     edgecolor="white", linewidth=1.5))
                ax.add_patch(Polygon([(j, i), (j + 1, i), (j + 1, i + 1)], facecolor=cface,
                                     edgecolor="white", linewidth=1.5))
                ax.plot([j, j + 1], [i, i + 1], color="white", linewidth=5)
                ax.text(j + .28, i + .72, f"{energy[i, i]:.3f}\n± {energy_sd[i, i]:.3f}",
                        ha="center", va="center", fontsize=11, color=readable_text(eface), fontweight="bold")
                ax.text(j + .72, i + .28, f"{correlation[i, i]:.2f}", ha="center", va="center",
                        fontsize=13, color=readable_text(cface), fontweight="bold")

    ax.set(xlim=(0, 5), ylim=(5, 0), aspect="equal")
    ax.set_xticks(np.arange(5) + .5, ELEMENTS, fontsize=16, fontweight="bold")
    ax.set_yticks(np.arange(5) + .5, ELEMENTS, fontsize=16, fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    divider = make_axes_locatable(ax)
    bottom = divider.append_axes("bottom", size="5%", pad=.55)
    right = divider.append_axes("right", size="5%", pad=.55)
    ebar = fig.colorbar(plt.cm.ScalarMappable(norm=energy_norm, cmap=energy_cmap), cax=bottom,
                        orientation="horizontal")
    ebar.set_label("Average Pair Interaction Energy (eV)", fontsize=15, fontweight="bold", labelpad=10)
    ebar.ax.tick_params(labelsize=13)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=corr_norm, cmap=corr_cmap), cax=right)
    cbar.set_label("Pearson Correlation: Pair Count vs Bulk Modulus",
                   fontsize=15, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=13)
    fig.savefig(HERE / "combined_publication_heatmap_corrected.png", dpi=800, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    table, mean, sd, corr = load_data()
    plot_scatter(table)
    plot_heatmap(mean, sd, corr)
    print("Created average_energy_vs_pair_count_correlation.png")
    print("Created combined_publication_heatmap_corrected.png")
