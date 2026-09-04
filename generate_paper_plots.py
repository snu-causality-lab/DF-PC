import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

results_dir = Path("results")
plots_dir = results_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
tex_path = results_dir / "supplementary_tables.tex"
tex_file = open(tex_path, "w")

# Style setup for premium publication look (aligned with main text plots)
mpl.rcParams.update({
    "font.family": "serif", 
    "font.size": 10,
    "axes.spines.top": False, 
    "axes.spines.right": False,
    "figure.dpi": 200, 
    "savefig.bbox": "tight", 
    "savefig.pad_inches": 0.05,
})

C_PC = "#888888"   # Gray for PC-stable
C_DD = "#fd8d3c"   # Orange for PC with deduce-dep
C_DF = "#2171b5"   # Blue for DF-PC

ALGO_PALETTE = {
    'PC-stable': C_PC,
    'PC with deduce-dep': C_DD,
    'DF-PC': C_DF
}

ALGO_MARKERS = {
    'PC-stable': 's',
    'PC with deduce-dep': '^',
    'DF-PC': 'o'
}

def rename_algo(df):
    df['Algo'] = df['Algo'].replace('DF-PC (Pure)', 'DF-PC')
    df['Algo'] = df['Algo'].replace('Standard PC', 'PC-stable')
    return df

def write_latex(df, title):
    tex_file.write(f"\\subsection*{{{title}}}\n")
    tex_file.write(df.to_latex(index=True, float_format="%.3f"))
    tex_file.write("\n\n")

def save_plots(fig, filename_base):
    # Save both png and pdf formats, under both expX and expY naming conventions for compatibility
    for ext in ["png", "pdf"]:
        name_map = {
            "exp7_": "exp1_",
            "exp8_": "exp2_",
            "exp9_": "exp3_",
            "exp10_": "exp4_",
            "exp11_": "exp5_"
        }
        fig.savefig(plots_dir / f"{filename_base}.{ext}", dpi=300)
        
        # Save alternative names
        alt_name = filename_base
        for k, v in name_map.items():
            if filename_base.startswith(k):
                alt_name = filename_base.replace(k, v)
                break
        if alt_name != filename_base:
            fig.savefig(plots_dir / f"{alt_name}.{ext}", dpi=300)

def find_csv(canonical_name, fallback_name):
    p1 = results_dir / canonical_name
    if p1.exists():
        return p1
    p2 = results_dir / fallback_name
    if p2.exists():
        return p2
    return None

# =========================================================================
# Exp 7: Baseline Comparison (DF-PC vs PC-stable vs Deduce-Dep)
# =========================================================================
f_sum = find_csv("experiment_baseline_comparison_summary.csv", "experiment_1_summary.csv")
f_raw = find_csv("experiment_baseline_comparison_raw.csv", "experiment_1_raw.csv")

if f_sum and f_raw:
    df1_sum = pd.read_csv(f_sum)
    df1_raw = pd.read_csv(f_raw)
    rename_algo(df1_sum); rename_algo(df1_raw)
    
    write_latex(df1_sum.groupby(['Algo', 'Data'])[['F1_mean', 'SHD_mean', 'Bypassed_CITs_mean']].mean(), "Experiment 7: Baseline Comparison")
 
    def plot_2x2(df_raw, metric, ylabel, filename_base):
        graphs = ['er', 'sf']
        dtypes = ['discrete', 'linear_sem']
        fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))
        for r, g in enumerate(graphs):
            for c, d in enumerate(dtypes):
                ax = axes[r][c]
                sub = df_raw[(df_raw['Graph'] == g) & (df_raw['Data'] == d)]
                sns.barplot(data=sub, x='Algo', y=metric, hue='Algo', palette=ALGO_PALETTE, 
                            order=['PC-stable', 'PC with deduce-dep', 'DF-PC'], ax=ax, legend=False,
                            edgecolor="white", errorbar=('ci', 95), capsize=0.1, err_kws={'linewidth': 1})
                ax.set_title(f"Graph={g.upper()}, Data={d.replace('_', ' ').title()}", fontsize=9, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel(ylabel, fontsize=9)
                ax.tick_params(axis='x', rotation=15, labelsize=8)
                ax.grid(True, axis='y', linestyle=":", alpha=0.5)
        fig.tight_layout()
        save_plots(fig, filename_base)
        plt.close('all')

    plot_2x2(df1_raw, 'F1', 'F1 Score', "exp7_f1")
    plot_2x2(df1_raw, 'Performed_CITs', 'Performed CITs', "exp7_performed_cits")

# =========================================================================
# Exp 8: High Density Regime (D=6)
# =========================================================================
f_sum = find_csv("experiment_high_density_summary.csv", "experiment_2_summary.csv")
f_raw = find_csv("experiment_high_density_raw.csv", "experiment_2_raw.csv")

if f_sum and f_raw:
    df2_sum = pd.read_csv(f_sum)
    df2_raw = pd.read_csv(f_raw)
    rename_algo(df2_sum); rename_algo(df2_raw)
    
    write_latex(df2_sum.groupby(['Algo', 'Data'])[['F1_mean', 'SHD_mean', 'Bypassed_CITs_mean']].mean(), "Experiment 8: High Density (=6)")

    def plot_2x2(df_raw, metric, ylabel, filename_base):
        graphs = ['er', 'sf']
        dtypes = ['discrete', 'linear_sem']
        fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))
        for r, g in enumerate(graphs):
            for c, d in enumerate(dtypes):
                ax = axes[r][c]
                sub = df_raw[(df_raw['Graph'] == g) & (df_raw['Data'] == d)]
                sns.barplot(data=sub, x='Algo', y=metric, hue='Algo', palette=ALGO_PALETTE,
                            order=['PC-stable', 'DF-PC'], ax=ax, legend=False,
                            edgecolor="white", errorbar=('ci', 95), capsize=0.1, err_kws={'linewidth': 1})
                ax.set_title(f"Graph={g.upper()}, Data={d.replace('_', ' ').title()}", fontsize=9, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel(ylabel, fontsize=9)
                ax.tick_params(axis='x', rotation=15, labelsize=8)
                ax.grid(True, axis='y', linestyle=":", alpha=0.5)
        fig.tight_layout()
        save_plots(fig, filename_base)
        plt.close('all')

    plot_2x2(df2_raw, 'F1', 'F1 Score', "exp8_f1")
    plot_2x2(df2_raw, 'Performed_CITs', 'Performed CITs', "exp8_performed_cits")

# =========================================================================
# Exp 9: Runtime Scaling across Nodes & Density
# =========================================================================
f_sum = find_csv("experiment_runtime_scaling_summary.csv", "experiment_3_summary.csv")
f_raw = find_csv("experiment_runtime_scaling_raw.csv", "experiment_3_raw.csv")

if f_sum and f_raw:
    df3_sum = pd.read_csv(f_sum)
    df3_raw = pd.read_csv(f_raw)
    rename_algo(df3_sum); rename_algo(df3_raw)
    
    write_latex(df3_sum.groupby(['Setup', 'Algo'])[['Time_Total_mean', 'Bypassed_CITs_mean', 'F1_mean']].mean(), "Experiment 9: Runtime Scaling")

    # Setup A: Node Scaling
    df3_nodes = df3_raw[df3_raw['Setup'] == 'SetupA_NodeScaling']
    graph_types = sorted(df3_nodes['Graph'].unique())
    fig, axes = plt.subplots(1, len(graph_types), figsize=(3.8 * len(graph_types), 3.2))
    if len(graph_types) == 1:
        axes = [axes]
    for i, (ax, gt) in enumerate(zip(axes, graph_types)):
        sub = df3_nodes[df3_nodes['Graph'] == gt]
        for algo in ['PC-stable', 'PC with deduce-dep', 'DF-PC']:
            s = sub[sub['Algo'] == algo]
            if s.empty: continue
            grouped = s.groupby('Nodes')['Time_Total'].agg(['mean', 'sem', 'count'])
            grouped['ci'] = grouped['sem'] * 1.96
            ax.plot(grouped.index, grouped['mean'], marker=ALGO_MARKERS[algo], color=ALGO_PALETTE[algo], label=algo, linewidth=1.5, markersize=5)
            ax.fill_between(grouped.index, grouped['mean'] - grouped['ci'], grouped['mean'] + grouped['ci'], alpha=0.15, color=ALGO_PALETTE[algo])
        ax.set_title(f"Graph={gt.upper()}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Number of Nodes (N)", fontsize=9)
        ax.set_ylabel('Total Runtime (s)', fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_plots(fig, "exp9_runtime_nodes")
    plt.close('all')

    # Setup B: Density Scaling
    df3_dens = df3_raw[df3_raw['Setup'] == 'SetupB_DensityScaling']
    graph_types = sorted(df3_dens['Graph'].unique())
    fig, axes = plt.subplots(1, len(graph_types), figsize=(3.8 * len(graph_types), 3.2))
    if len(graph_types) == 1:
        axes = [axes]
    for i, (ax, gt) in enumerate(zip(axes, graph_types)):
        sub = df3_dens[df3_dens['Graph'] == gt]
        for algo in ['PC-stable', 'PC with deduce-dep', 'DF-PC']:
            s = sub[sub['Algo'] == algo]
            if s.empty: continue
            grouped = s.groupby('Deg')['Time_Total'].agg(['mean', 'sem', 'count'])
            grouped['ci'] = grouped['sem'] * 1.96
            ax.plot(grouped.index, grouped['mean'], marker=ALGO_MARKERS[algo], color=ALGO_PALETTE[algo], label=algo, linewidth=1.5, markersize=5)
            ax.fill_between(grouped.index, grouped['mean'] - grouped['ci'], grouped['mean'] + grouped['ci'], alpha=0.15, color=ALGO_PALETTE[algo])
        ax.set_title(f"Graph={gt.upper()}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Average Degree (D)", fontsize=9)
        ax.set_ylabel('Total Runtime (s)', fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_plots(fig, "exp9_runtime_density")
    plt.close('all')

# =========================================================================
# Exp 10: Oracle Saved CITs Breakdown by |Z|
# =========================================================================
f_sum = find_csv("experiment_oracle_saved_cits_summary.csv", "experiment_4_summary.csv")
f_raw = find_csv("experiment_oracle_saved_cits_raw.csv", "experiment_4_raw.csv")

if f_sum and f_raw:
    df4_sum = pd.read_csv(f_sum)
    df4_raw = pd.read_csv(f_raw)
    
    write_latex(df4_sum.groupby(['Setup', 'Z_Size'])[['Requested_mean', 'Performed_mean', 'Bypassed_mean']].mean(), "Experiment 10: Oracle Saved CITs by |Z|")

    def plot_exp4_faceted(df_raw_setup, facet_col, facet_vals, filename_base):
        df_f = df_raw_setup[(df_raw_setup['Z_Size'] > 0) & (df_raw_setup['Z_Size'] <= 8)].copy()
        n_facets = len(facet_vals)
        fig, axes = plt.subplots(1, n_facets, figsize=(3.8 * n_facets, 3.2))
        if n_facets == 1:
            axes = [axes]
        axes_flat = axes.flatten()
        
        c_req = "#bdbdbd"
        c_perf = "#2171b5"
        c_by = "#9ecae1"
        c_palette = {'Requested CITs': c_req, 'Performed CITs': c_perf, 'Bypassed CITs': c_by}
        
        for i, (ax, val) in enumerate(zip(axes_flat, facet_vals)):
            sub = df_f[df_f[facet_col] == val]
            sub_m = sub.melt(id_vars=['Z_Size'], value_vars=['Requested_CITs', 'Performed_CITs', 'Bypassed_CITs'], var_name='Type', value_name='Count')
            sub_m['Type'] = sub_m['Type'].map({'Requested_CITs': 'Requested CITs', 'Performed_CITs': 'Performed CITs', 'Bypassed_CITs': 'Bypassed CITs'})
            
            sns.barplot(data=sub_m, x='Z_Size', y='Count', hue='Type', palette=c_palette, ax=ax,
                        edgecolor="white", errorbar=('ci', 95), capsize=0.1, err_kws={'linewidth': 1})
            ax.set_title(f"{facet_col}={val}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Conditioning Set Size ($|Z|$)", fontsize=9)
            ax.set_ylabel("Number of CITs", fontsize=9)
            ax.grid(True, axis='y', linestyle=":", alpha=0.5)
            if i == 0:
                ax.legend(frameon=False, fontsize=8)
            else:
                try: ax.legend_.remove()
                except: pass

        fig.tight_layout()
        save_plots(fig, filename_base)
        plt.close('all')

    # Setup A: Node Scaling
    df4_A = df4_raw[df4_raw['Setup'] == 'SetupA_Nodes']
    node_vals = sorted(df4_A['Nodes'].unique())
    plot_exp4_faceted(df4_A, 'Nodes', node_vals, "exp10_save_CIT_nodes")

    # Setup B: Density Scaling
    df4_B = df4_raw[df4_raw['Setup'] == 'SetupB_Density']
    deg_vals = sorted(df4_B['Deg'].unique())
    plot_exp4_faceted(df4_B, 'Deg', deg_vals, "exp10_save_CIT_density")

# =========================================================================
# Exp 11: Algorithmic Resilience to CIT Errors
# =========================================================================
f_sum = find_csv("experiment_noise_robustness_summary.csv", "experiment_5_summary.csv")
f_raw = find_csv("experiment_noise_robustness_raw.csv", "experiment_5_raw.csv")

if f_sum and f_raw:
    df5_sum = pd.read_csv(f_sum)
    df5_raw = pd.read_csv(f_raw)
    rename_algo(df5_sum); rename_algo(df5_raw)
    
    write_latex(df5_sum.groupby(['ErrorRate', 'Algo'])[['F1_mean', 'SHD_mean']].mean(), "Experiment 11: Robustness against Low-order CIT Error")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for algo in ['PC-stable', 'PC with deduce-dep', 'DF-PC']:
        s = df5_raw[df5_raw['Algo'] == algo]
        if s.empty: continue
        grouped = s.groupby('ErrorRate')['F1'].agg(['mean', 'sem'])
        grouped['ci'] = grouped['sem'] * 1.96
        ax.plot(grouped.index, grouped['mean'], marker=ALGO_MARKERS[algo], color=ALGO_PALETTE[algo], label=algo, linewidth=1.5, markersize=5)
        ax.fill_between(grouped.index, grouped['mean'] - grouped['ci'], grouped['mean'] + grouped['ci'], alpha=0.15, color=ALGO_PALETTE[algo])
        
    ax.set_xlabel("Error Rate ($p$)", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_plots(fig, "exp11_robustness")
    plt.close()

tex_file.close()
print("Plots generated successfully in results/plots/ and LaTeX tables saved to results/supplementary_tables.tex.")
