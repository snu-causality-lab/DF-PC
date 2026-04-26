import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

folder = "results_rebuttal"
os.makedirs(f"{folder}/plots", exist_ok=True)
tex_file = open(f"{folder}/rebuttal_tables.tex", "w")

def rename_algo(df):
    df['Algo'] = df['Algo'].replace('DF-PC (Pure)', 'DF-PC')
    return df

def write_latex(df, title):
    tex_file.write(f"\\subsection*{{{title}}}\n")
    tex_file.write(df.to_latex(index=True, float_format="%.3f"))
    tex_file.write("\n\n")

# Exp 1
if os.path.exists(f"{folder}/experiment_1_summary.csv") and os.path.exists(f"{folder}/experiment_1_raw.csv"):
    df1_sum = pd.read_csv(f"{folder}/experiment_1_summary.csv")
    df1_raw = pd.read_csv(f"{folder}/experiment_1_raw.csv")
    rename_algo(df1_sum); rename_algo(df1_raw)
    
    write_latex(df1_sum.groupby(['Algo', 'Data'])[['F1_mean', 'SHD_mean', 'Bypassed_CITs_mean']].mean(), "Experiment 1: Baseline Comparison")

    def plot_2x2(df_raw, metric, ylabel, suptitle, filename):
        graphs = ['er', 'sf']
        dtypes = ['discrete', 'linear_sem']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for r, g in enumerate(graphs):
            for c, d in enumerate(dtypes):
                ax = axes[r][c]
                sub = df_raw[(df_raw['Graph'] == g) & (df_raw['Data'] == d)]
                sns.barplot(data=sub, x='Algo', y=metric, hue='Algo', capsize=.1, ax=ax, legend=False)
                ax.set_title(f"Graph={g.upper()}, Data={d}")
                ax.set_xlabel("")
                ax.set_ylabel(ylabel)
                ax.tick_params(axis='x', rotation=15)
        fig.suptitle(suptitle, y=1.02)
        fig.tight_layout()
        fig.savefig(f"{folder}/plots/{filename}", dpi=300, bbox_inches='tight')
        plt.close('all')

    plot_2x2(df1_raw, 'F1', 'F1 Score',
             "Exp 1: F1 Score Comparison\n[Nodes=20, Samples=5000, Degree=4]",
             "exp1_f1.png")
    plot_2x2(df1_raw, 'Performed_CITs', 'Performed CITs',
             "Exp 1: Performed CITs\n[Nodes=20, Samples=5000, Degree=4]",
             "exp1_performed_cits.png")


# Exp 2
if os.path.exists(f"{folder}/experiment_2_summary.csv") and os.path.exists(f"{folder}/experiment_2_raw.csv"):
    df2_sum = pd.read_csv(f"{folder}/experiment_2_summary.csv")
    df2_raw = pd.read_csv(f"{folder}/experiment_2_raw.csv")
    rename_algo(df2_sum); rename_algo(df2_raw)
    
    write_latex(df2_sum.groupby(['Algo', 'Data'])[['F1_mean', 'SHD_mean', 'Bypassed_CITs_mean']].mean(), "Experiment 2: High Density (=6)")

    def plot_2x2(df_raw, metric, ylabel, suptitle, filename):
        graphs = ['er', 'sf']
        dtypes = ['discrete', 'linear_sem']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for r, g in enumerate(graphs):
            for c, d in enumerate(dtypes):
                ax = axes[r][c]
                sub = df_raw[(df_raw['Graph'] == g) & (df_raw['Data'] == d)]
                sns.barplot(data=sub, x='Algo', y=metric, hue='Algo', capsize=.1, ax=ax, legend=False)
                ax.set_title(f"Graph={g.upper()}, Data={d}")
                ax.set_xlabel("")
                ax.set_ylabel(ylabel)
                ax.tick_params(axis='x', rotation=15)
        fig.suptitle(suptitle, y=1.02)
        fig.tight_layout()
        fig.savefig(f"{folder}/plots/{filename}", dpi=300, bbox_inches='tight')
        plt.close('all')

    plot_2x2(df2_raw, 'F1', 'F1 Score',
             "Exp 2: F1 Score Comparison\n[Nodes=30, Samples=5000, Degree=6]",
             "exp2_f1.png")
    plot_2x2(df2_raw, 'Performed_CITs', 'Performed CITs',
             "Exp 2: Performed CITs\n[Nodes=30, Samples=5000, Degree=6]",
             "exp2_performed_cits.png")


# Exp 3
if os.path.exists(f"{folder}/experiment_3_summary.csv") and os.path.exists(f"{folder}/experiment_3_raw.csv"):
    df3_sum = pd.read_csv(f"{folder}/experiment_3_summary.csv")
    df3_raw = pd.read_csv(f"{folder}/experiment_3_raw.csv")
    rename_algo(df3_sum); rename_algo(df3_raw)
    
    write_latex(df3_sum.groupby(['Setup', 'Algo'])[['Time_Total_mean', 'Bypassed_CITs_mean', 'F1_mean']].mean(), "Experiment 3: Runtime Scaling")

    # Setup A: Node Scaling — faceted by Graph
    df3_nodes = df3_raw[df3_raw['Setup'] == 'SetupA_NodeScaling']
    graph_types = sorted(df3_nodes['Graph'].unique())
    fig, axes = plt.subplots(1, len(graph_types), figsize=(7 * len(graph_types), 5))
    if len(graph_types) == 1:
        axes = [axes]
    for i, (ax, gt) in enumerate(zip(axes, graph_types)):
        sub = df3_nodes[df3_nodes['Graph'] == gt]
        sns.lineplot(data=sub, x='Nodes', y='Time_Total', hue='Algo', marker='o', err_style='bars', ax=ax)
        ax.set_title(f"Graph={gt.upper()}")
        ax.set_ylabel('Total Runtime (s)')
        if i > 0:
            try: ax.legend_.remove()
            except: pass
    fig.suptitle("Exp 3 Setup A: Runtime vs Nodes\n[Samples=5000, Degree=3, Data=LinearSEM]", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{folder}/plots/exp3_runtime_nodes.png", dpi=300, bbox_inches='tight')
    plt.close('all')

    # Setup B: Density Scaling — faceted by Graph
    df3_dens = df3_raw[df3_raw['Setup'] == 'SetupB_DensityScaling']
    graph_types = sorted(df3_dens['Graph'].unique())
    fig, axes = plt.subplots(1, len(graph_types), figsize=(7 * len(graph_types), 5))
    if len(graph_types) == 1:
        axes = [axes]
    for i, (ax, gt) in enumerate(zip(axes, graph_types)):
        sub = df3_dens[df3_dens['Graph'] == gt]
        sns.lineplot(data=sub, x='Deg', y='Time_Total', hue='Algo', marker='o', err_style='bars', ax=ax)
        ax.set_title(f"Graph={gt.upper()}")
        ax.set_ylabel('Total Runtime (s)')
        if i > 0:
            try: ax.legend_.remove()
            except: pass
    fig.suptitle("Exp 3 Setup B: Runtime vs Density\n[Nodes=30, Samples=5000, Data=LinearSEM]", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{folder}/plots/exp3_runtime_density.png", dpi=300, bbox_inches='tight')
    plt.close('all')

# Exp 4
if os.path.exists(f"{folder}/experiment_4_summary.csv") and os.path.exists(f"{folder}/experiment_4_raw.csv"):
    df4_sum = pd.read_csv(f"{folder}/experiment_4_summary.csv")
    df4_raw = pd.read_csv(f"{folder}/experiment_4_raw.csv")
    
    write_latex(df4_sum.groupby(['Setup', 'Z_Size'])[['Requested_mean', 'Performed_mean', 'Bypassed_mean']].mean(), "Experiment 4: Oracle Saved CITs by |Z|")

    def plot_exp4_faceted(df_raw_setup, facet_col, facet_vals, suptitle, filename):
        df_f = df_raw_setup[(df_raw_setup['Z_Size'] > 0) & (df_raw_setup['Z_Size'] <= 8)].copy()
        n_facets = len(facet_vals)
        nrows = 2
        ncols = (n_facets + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes_flat = axes.flatten()
        for i, (ax, val) in enumerate(zip(axes_flat, facet_vals)):
            sub = df_f[df_f[facet_col] == val]
            sub_m = sub.melt(id_vars=['Z_Size'], value_vars=['Requested_CITs', 'Performed_CITs', 'Bypassed_CITs'], var_name='Type', value_name='Count')
            sub_m['Type'] = sub_m['Type'].map({'Requested_CITs': 'Requested', 'Performed_CITs': 'Performed', 'Bypassed_CITs': 'Bypassed'})
            sns.barplot(data=sub_m, x='Z_Size', y='Count', hue='Type', capsize=.05, ax=ax)
            ax.set_title(f"{facet_col}={val}")
            ax.set_xlabel("Conditioning Set Size (|Z|)")
            ax.set_ylabel("Number of CITs")
            if i > 0:
                try:
                    ax.legend_.remove()
                except:
                    pass
        # Hide unused axes
        for j in range(len(facet_vals), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(suptitle, y=1.02)
        fig.tight_layout()
        fig.savefig(f"{folder}/plots/{filename}", dpi=300, bbox_inches='tight')
        plt.close('all')

    # Setup A: Node Scaling — one subplot per Node count (2x2)
    df4_A = df4_raw[df4_raw['Setup'] == 'SetupA_Nodes']
    node_vals = sorted(df4_A['Nodes'].unique())
    plot_exp4_faceted(df4_A, 'Nodes', node_vals,
                      "Exp 4 Setup A: CITs by |Z| across Node Counts\n[Degree=3, Topology=ER, Oracle CIT]",
                      "exp4_save_CIT_nodes.png")

    # Setup B: Density Scaling — one subplot per Degree (2x2)
    df4_B = df4_raw[df4_raw['Setup'] == 'SetupB_Density']
    deg_vals = sorted(df4_B['Deg'].unique())
    plot_exp4_faceted(df4_B, 'Deg', deg_vals,
                      "Exp 4 Setup B: CITs by |Z| across Density Levels\n[Nodes=20, Topology=ER, Oracle CIT]",
                      "exp4_save_CIT_density.png")

# Exp 5
if os.path.exists(f"{folder}/experiment_5_summary.csv") and os.path.exists(f"{folder}/experiment_5_raw.csv"):
    df5_sum = pd.read_csv(f"{folder}/experiment_5_summary.csv")
    df5_raw = pd.read_csv(f"{folder}/experiment_5_raw.csv")
    rename_algo(df5_sum); rename_algo(df5_raw)
    
    write_latex(df5_sum.groupby(['ErrorRate', 'Algo'])[['F1_mean', 'SHD_mean']].mean(), "Experiment 5: Robustness against Low-order CIT Error")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df5_raw, x='ErrorRate', y='F1', hue='Algo', marker='o', err_style="bars")
    plt.title("Exp 5: Robustness — F1 vs Oracle CIT Error Rate\n[Nodes=20, Degree=3, Topology=ER, Oracle d-sep]\nNoise: low-order CITs (|Z|≤1) flipped with prob p via deterministic blake2b hash")
    plt.xlabel("Error Rate (p)")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(f"{folder}/plots/exp5_robustness.png", dpi=300)
    plt.close()

tex_file.close()
print("SUCCESS")
