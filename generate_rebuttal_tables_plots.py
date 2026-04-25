import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

folder = "results_rebuttal"
os.makedirs(f"{folder}/plots", exist_ok=True)
tex_file = open(f"{folder}/rebuttal_tables.tex", "w")

def write_latex(df, title):
    tex_file.write(f"\\subsection*{{{title}}}\n")
    tex_file.write(df.to_latex(index=True, float_format="%.3f"))
    tex_file.write("\n\n")

# Exp 1
if os.path.exists(f"{folder}/experiment_1_summary.csv") and os.path.exists(f"{folder}/experiment_1_raw.csv"):
    df1_sum = pd.read_csv(f"{folder}/experiment_1_summary.csv")
    df1_raw = pd.read_csv(f"{folder}/experiment_1_raw.csv")
    
    write_latex(df1_sum.groupby(['Algo', 'Data'])[['F1_mean', 'SHD_mean', 'Bypassed_CITs_mean', 'Time_Total_mean']].mean(), "Experiment 1: Baseline Comparison")

    # Plot F1
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df1_raw, x='Data', y='F1', hue='Algo', capsize=.1)
    plt.title("Exp 1: F1 Score Comparison")
    plt.savefig(f"{folder}/plots/exp1_f1.png", dpi=300)
    plt.close()

    # Plot CITs (Performed)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df1_raw, x='Data', y='Performed_CITs', hue='Algo', capsize=.1)
    plt.title("Exp 1: Performed CITs")
    plt.savefig(f"{folder}/plots/exp1_performed_cits.png", dpi=300)
    plt.close()

    # Plot Time
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df1_raw, x='Data', y='Time_Total', hue='Algo', capsize=.1)
    plt.title("Exp 1: Total Runtime (Seconds)")
    plt.savefig(f"{folder}/plots/exp1_time.png", dpi=300)
    plt.close()

# Exp 2
if os.path.exists(f"{folder}/experiment_2_summary.csv") and os.path.exists(f"{folder}/experiment_2_raw.csv"):
    df2_sum = pd.read_csv(f"{folder}/experiment_2_summary.csv")
    df2_raw = pd.read_csv(f"{folder}/experiment_2_raw.csv")
    
    write_latex(df2_sum.groupby(['Algo', 'Data'])[['F1_mean', 'SHD_mean', 'Bypassed_CITs_mean', 'Time_Total_mean']].mean(), "Experiment 2: High Density (=6)")

    # Plot F1
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df2_raw, x='Data', y='F1', hue='Algo', capsize=.1)
    plt.title("Exp 2: F1 Score Comparison (Density=6)")
    plt.savefig(f"{folder}/plots/exp2_f1.png", dpi=300)
    plt.close()

    # Plot CITs
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df2_raw, x='Data', y='Performed_CITs', hue='Algo', capsize=.1)
    plt.title("Exp 2: Performed CITs (Density=6)")
    plt.savefig(f"{folder}/plots/exp2_performed_cits.png", dpi=300)
    plt.close()

    # Plot Time
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df2_raw, x='Data', y='Time_Total', hue='Algo', capsize=.1)
    plt.title("Exp 2: Total Runtime (Density=6)")
    plt.savefig(f"{folder}/plots/exp2_time.png", dpi=300)
    plt.close()

# Exp 3
if os.path.exists(f"{folder}/experiment_3_summary.csv") and os.path.exists(f"{folder}/experiment_3_raw.csv"):
    df3_sum = pd.read_csv(f"{folder}/experiment_3_summary.csv")
    df3_raw = pd.read_csv(f"{folder}/experiment_3_raw.csv")
    
    write_latex(df3_sum.groupby(['Setup', 'Algo'])[['Time_Total_mean', 'Bypassed_CITs_mean', 'F1_mean']].mean(), "Experiment 3: Runtime Scaling")

    # Setup A (Nodes)
    df3_nodes = df3_raw[df3_raw['Setup'] == 'SetupA_NodeScaling']
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df3_nodes, x='Nodes', y='Time_Total', hue='Algo', marker='o', err_style="bars")
    plt.title("Exp 3: Runtime vs Nodes (Density=3)")
    plt.savefig(f"{folder}/plots/exp3_runtime_nodes.png", dpi=300)
    plt.close()

    # Setup B (Density)
    df3_dens = df3_raw[df3_raw['Setup'] == 'SetupB_DensityScaling']
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df3_dens, x='Deg', y='Time_Total', hue='Algo', marker='o', err_style="bars")
    plt.title("Exp 3: Runtime vs Density (Nodes=30)")
    plt.savefig(f"{folder}/plots/exp3_runtime_density.png", dpi=300)
    plt.close()

# Exp 4
if os.path.exists(f"{folder}/experiment_4_summary.csv") and os.path.exists(f"{folder}/experiment_4_raw.csv"):
    df4_sum = pd.read_csv(f"{folder}/experiment_4_summary.csv")
    df4_raw = pd.read_csv(f"{folder}/experiment_4_raw.csv")
    
    df4_setupB = df4_sum[df4_sum['Setup'] == 'SetupB_Density']
    write_latex(df4_setupB.groupby('Z_Size')[['Requested_mean', 'Performed_mean', 'Bypassed_mean']].mean(), "Experiment 4: Oracle Saved CITs by |Z| (Setup B)")

    df4_raw_B = df4_raw[df4_raw['Setup'] == 'SetupB_Density']
    # Filter out Z=0
    df4_raw_B = df4_raw_B[df4_raw_B['Z_Size'] > 0]
    # Filter trailing zeros (where Requested is dynamically non-existent, just keep Z_Size <= 8)
    df4_raw_B = df4_raw_B[df4_raw_B['Z_Size'] <= 8]

    df4_melted = df4_raw_B.melt(id_vars=['Z_Size'], value_vars=['Requested_CITs', 'Performed_CITs', 'Bypassed_CITs'], var_name='CIT Type', value_name='Count')
    # Use full names for legend
    df4_melted['CIT Type'] = df4_melted['CIT Type'].map({
        'Requested_CITs': 'Requested CITs', 
        'Performed_CITs': 'Performed CITs', 
        'Bypassed_CITs': 'Bypassed CITs'
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df4_melted, x='Z_Size', y='Count', hue='CIT Type', capsize=.05)
    plt.title("Exp 4: CITs by Conditioning Set Size |Z|")
    plt.ylabel("Number of CITs")
    plt.xlabel("Conditioning Set Size (|Z|)")
    plt.tight_layout()
    plt.savefig(f"{folder}/plots/exp4_cits_by_z.png", dpi=300)
    plt.close()

# Exp 5
if os.path.exists(f"{folder}/experiment_5_summary.csv") and os.path.exists(f"{folder}/experiment_5_raw.csv"):
    df5_sum = pd.read_csv(f"{folder}/experiment_5_summary.csv")
    df5_raw = pd.read_csv(f"{folder}/experiment_5_raw.csv")
    
    write_latex(df5_sum.groupby(['ErrorRate', 'Algo'])[['F1_mean', 'SHD_mean']].mean(), "Experiment 5: Robustness against Low-order CIT Error")

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df5_raw, x='ErrorRate', y='F1', hue='Algo', marker='o', err_style="bars")
    plt.title("Exp 5: Robustness (F1 Score Drop over Error Rate)")
    plt.savefig(f"{folder}/plots/exp5_robustness.png", dpi=300)
    plt.close()

tex_file.close()
print("SUCCESS")
