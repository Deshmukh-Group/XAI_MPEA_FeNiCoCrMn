import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, Polygon

# ---------------- SETTINGS ----------------
ENERGY_CSV = "per_file_pair_summary.csv"
CORR_CSV = "1final_matched_summary.csv"
OUTPUT_FILE = "combined_publication_heatmap.png"
TYPE_MAP = {1: 'Mn', 2: 'Cr', 3: 'Co', 4: 'Fe', 5: 'Ni'}
# ------------------------------------------

def get_text_color(bg_color):
    """Calculate luminance to determine whether text should be black or white for optimal contrast."""
    r, g, b = bg_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.55 else 'black'

def generate_publication_plot():
    # 1. Load Data
    print("Loading data...")
    df_energy = pd.read_csv(ENERGY_CSV)
    df_corr = pd.read_csv(CORR_CSV)

    # 2. Process Energy Data
    df_energy['Total_Energy_Contribution'] = df_energy['Count'] * df_energy['Mean_Energy']
    pair_stats = df_energy.groupby('Pair').agg(
        Sum_Energy=('Total_Energy_Contribution', 'sum'),
        Total_Count=('Count', 'sum'),
        Std_File_Mean=('Mean_Energy', 'std')
    ).reset_index()
    pair_stats['Weighted_Mean_Energy'] = pair_stats['Sum_Energy'] / pair_stats['Total_Count']

    elements = ["Mn", "Cr", "Co", "Fe", "Ni"]
    n = len(elements)
    
    # Initialize matrices
    energy_matrix = np.zeros((n, n))
    energy_std_matrix = np.zeros((n, n))
    corr_matrix = np.zeros((n, n))

    print("Building matrices...")
    for i in range(n):
        for j in range(n):
            e1_idx = i + 1
            e2_idx = j + 1
            
            # Energy Data (Symmetric)
            p_min, p_max = sorted((e1_idx, e2_idx))
            pair_str = f"{p_min}-{p_max}"
            
            row = pair_stats[pair_stats['Pair'] == pair_str]
            if not row.empty:
                energy_matrix[i, j] = row['Weighted_Mean_Energy'].values[0]
                energy_std_matrix[i, j] = row['Std_File_Mean'].values[0]
            
            # Correlation Data (Symmetric)
            col_name = f"N_{p_min}-{p_max}"
            if col_name in df_corr.columns:
                from scipy import stats
                r, _ = stats.pearsonr(df_corr[col_name], df_corr['Y_Actual'])
                corr_matrix[i, j] = r

    # 3. Setup Colormaps (Improved Contrast)
    # Energy: 'cividis' 
    cmap_energy = plt.cm.viridis_r
    norm_energy = mcolors.Normalize(vmin=np.min(energy_matrix), vmax=np.max(energy_matrix))
    
    # Correlation: 'RdBu_r' 
    cmap_corr = plt.get_cmap("RdBu_r")
    norm_corr = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # 4. Plotting
    print("Generating plot...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(figsize=(12, 12))

    for i in range(n):
        for j in range(n):
            x = j
            y = i
            
            if i > j:
                # Lower Triangle: Energy
                val = energy_matrix[i, j]
                std = energy_std_matrix[i, j]
                
                bg_color = cmap_energy(norm_energy(val))
                rect = Rectangle((x, y), 1, 1, facecolor=bg_color, edgecolor='white', linewidth=1.5)
                ax.add_patch(rect)
                
                text_color = get_text_color(bg_color)
                ax.text(x + 0.5, y + 0.5, f"{val:.3f}\n± {std:.3f}",
                        ha='center', va='center', fontsize=14, color=text_color, fontweight='bold')

            elif i < j:
                # Upper Triangle: Correlation
                val = corr_matrix[i, j]
                
                bg_color = cmap_corr(norm_corr(val))
                rect = Rectangle((x, y), 1, 1, facecolor=bg_color, edgecolor='white', linewidth=1.5)
                ax.add_patch(rect)
                
                text_color = get_text_color(bg_color)
                ax.text(x + 0.5, y + 0.5, f"{val:.2f}",
                        ha='center', va='center', fontsize=15, color=text_color, fontweight='bold')

            else:
                # Diagonal: Split
                e_val = energy_matrix[i, i]
                e_std = energy_std_matrix[i, i]
                c_val = corr_matrix[i, i]
                
                e_color = cmap_energy(norm_energy(e_val))
                c_color = cmap_corr(norm_corr(c_val))

                # Diagonal line from bottom-left (x, y+1) to top-right (x+1, y)
                tri_energy = Polygon([(x, y+1), (x+1, y+1), (x, y)], facecolor=e_color, edgecolor='white', linewidth=1.5)
                ax.add_patch(tri_energy)
                
                tri_corr = Polygon([(x, y), (x+1, y), (x+1, y+1)], facecolor=c_color, edgecolor='white', linewidth=1.5)
                ax.add_patch(tri_corr)
                
                # Add a white diagonal line to cover the seam properly
                ax.plot([x, x+1], [y, y+1], color='white', linewidth=5)
                
                # Text
                e_text_color = get_text_color(e_color)
                ax.text(x + 0.28, y + 0.72, f"{e_val:.3f}\n± {e_std:.3f}",
                        ha='center', va='center', fontsize=11, color=e_text_color, fontweight='bold')
                
                c_text_color = get_text_color(c_color)
                ax.text(x + 0.72, y + 0.28, f"{c_val:.2f}",
                        ha='center', va='center', fontsize=13, color=c_text_color, fontweight='bold')


    # 5. Formatting
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(elements, fontsize=16, fontweight='bold')
    ax.set_yticklabels(elements, fontsize=16, fontweight='bold')

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.tick_params(axis='both', which='both', length=0)

    # 6. Colorbars
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    
    cax_bottom = divider.append_axes("bottom", size="5%", pad=0.5)
    cax_right = divider.append_axes("right", size="5%", pad=0.5)

    sm_energy = plt.cm.ScalarMappable(norm=norm_energy, cmap=cmap_energy)
    cb_energy = plt.colorbar(sm_energy, cax=cax_bottom, orientation='horizontal')
    cb_energy.set_label('Average Pair Interaction Energy (eV)', fontsize=16, fontweight='bold', labelpad=10)
    cb_energy.ax.tick_params(labelsize=14)

    sm_corr = plt.cm.ScalarMappable(norm=norm_corr, cmap=cmap_corr)
    cb_corr = plt.colorbar(sm_corr, cax=cax_right, orientation='vertical')
    cb_corr.set_label('Pearson Correlation', fontsize=16, fontweight='bold', labelpad=10)
    cb_corr.ax.tick_params(labelsize=14)

    #fig.suptitle("Combined Matrix: Pair Energy vs Bulk Modulus Correlation", fontsize=20, fontweight='bold', y=1.05)

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Publication figure saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_publication_plot()
