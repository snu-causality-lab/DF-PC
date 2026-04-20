import pandas as pd
import numpy as np
import os
from pathlib import Path

def format_cell(mean, ci=None, is_best=False, lower_is_better=False):
    if pd.isna(mean):
        return "-"
    if ci is not None:
        val_str = f"{mean:.2f}\\stdv{{{ci:.2f}}}"
        if is_best:
            return f"\\textbf{{{mean:.2f}}}\\stdv{{{ci:.2f}}}"
        return val_str
    else:
        val_str = f"{mean:.2f}"
        if is_best:
            return f"\\textbf{{{mean:.2f}}}"
        return val_str

def gen_synthetic_benchmarks(df, out):
    """Experiment 1: Synthetic Benchmarks (N=10, 20, 30)"""
    for n in sorted(df['Nodes'].unique()):
        df_n = df[df['Nodes'] == n]
        if len(df_n) == 0: continue
        
        # Performance Table
        out.append(f"% --- Experiment 1: Performance (N={n}) ---")
        out.append(r"\begin{table*}[t!]\centering")
        out.append(f"\\caption{{Learning Performance for $|\\mathbf{{V}}|={n}$.}}\\label{{table:perf-n{n}}}")
        out.append(r"\begin{adjustbox}{width=\linewidth}\begin{tabular}{@{} c c c l c c c c c c c c @{}}\toprule")
        out.append(r"\multirow{2}{*}{Graph} & \multirow{2}{*}{$D$} & \multirow{2}{*}{$M$} & \multirow{2}{*}{Method} & \multicolumn{4}{c}{Discrete} & \multicolumn{4}{c}{Linear SEM} \\")
        out.append(r"\cmidrule(lr){5-8} \cmidrule(l){9-12} & & & & F1 & Prec. & Rec. & SHD & F1 & Prec. & Rec. & SHD \\\midrule")
        
        for g in ['er', 'sf']:
            for d in sorted(df_n['Deg'].unique()):
                samples = sorted(df_n['Samples'].unique())
                for m_idx, m in enumerate(samples):
                    for a_idx, algo in enumerate(["Standard PC", "DF-PC (Pure)"]):
                        prefix = []
                        if d == sorted(df_n['Deg'].unique())[0] and m_idx == 0 and a_idx == 0:
                            prefix.append(f"\\multirow{{16}}{{*}}{{\\rotatebox{{90}}{{{g.upper()}}}}}")
                        else: prefix.append("")
                        if m_idx == 0 and a_idx == 0: prefix.append(f"\\multirow{{8}}{{*}}{{{d}}}")
                        else: prefix.append("")
                        if a_idx == 0: prefix.append(f"\\multirow{{2}}{{*}}{{{m}}}")
                        else: prefix.append("")
                        algo_lbl = "Standard PC" if algo == "Standard PC" else r"\textbf{DF-PC}"
                        prefix.append(algo_lbl)
                        
                        row_str = " & ".join(prefix)
                        for dt in ['discrete', 'linear_sem']:
                            sub = df_n[(df_n['Graph']==g) & (df_n['Deg']==d) & (df_n['Samples']==m) & (df_n['Data']==dt)]
                            best_f1 = sub['F1_mean'].max() if not sub.empty else -1
                            best_shd = sub['SHD_mean'].min() if not sub.empty else float('inf')
                            r_data = sub[sub['Algo'] == algo]
                            if not r_data.empty:
                                r = r_data.iloc[0]
                                f1 = format_cell(r['F1_mean'], r['F1_ci95'], r['F1_mean'] >= best_f1 - 1e-4)
                                pr = format_cell(r['Prec_mean'], r['Prec_ci95'], False)
                                re = format_cell(r['Rec_mean'], r['Rec_ci95'], False)
                                sh = format_cell(r['SHD_mean'], r['SHD_ci95'], r['SHD_mean'] <= best_shd + 1e-4, True)
                                row_str += f" & {f1} & {pr} & {re} & {sh}"
                            else: row_str += " & - & - & - & -"
                        out.append(row_str + r" \\")
                out.append(r"\cmidrule(lr){2-12}")
            if g == 'er': out.append(r"\midrule")
        out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table*}" + "\n")

def gen_synthetic_efficiency(df, out):
    """Experiment 1: Computational Efficiency (N=10, 20, 30)"""
    for n in sorted(df['Nodes'].unique()):
        df_n = df[df['Nodes'] == n]
        if len(df_n) == 0: continue
        
        out.append(f"% --- Experiment 1: Efficiency (N={n}) ---")
        out.append(r"\begin{table}[h!]\centering")
        out.append(f"\\caption{{Computational Efficiency for $|\\mathbf{{V}}|={n}$. \\textit{{Cond. CIT}} denotes conditional independence tests ($|Z|>0$), \\textit{{Req.}} is the total CI requests from the framework, and \\textit{{OH Time}} is the algorithmic overhead in seconds.}}\\label{{table:eff-n{n}}}")
        out.append(r"\begin{adjustbox}{width=\columnwidth}\begin{tabular}{@{} c c l c c c c c @{}}\toprule")
        out.append(r"Graph & $D$ & Method & Cond. CIT & Req. & Tot. Time & OH Time \\ \midrule")
        
        for g in ['er', 'sf']:
            for d in sorted(df_n['Deg'].unique()):
                sub = df_n[(df_n['Graph']==g) & (df_n['Deg']==d) & (df_n['Samples']==5000) & (df_n['Data']=='discrete')]
                if sub.empty: continue
                
                for i, algo in enumerate(["Standard PC", "DF-PC (Pure)"]):
                    r_data = sub[sub['Algo'] == algo]
                    if r_data.empty: continue
                    r = r_data.iloc[0]
                    
                    prefix = f"\\multirow{{2}}{{*}}{{{g.upper()}}}" if i == 0 and d == sorted(df_n['Deg'].unique())[0] else ""
                    deg_prefix = f"\\multirow{{2}}{{*}}{{{d}}}" if i == 0 else ""
                    algo_lbl = "Standard PC" if algo == "Standard PC" else r"\textbf{DF-PC}"
                    
                    cond = format_cell(r['Cond_CIT_mean'], r['Cond_CIT_ci95'])
                    req = format_cell(r['Requests_mean'], r['Requests_ci95'])
                    tot = format_cell(r['Time_Total_mean'], r['Time_Total_ci95'])
                    oh = format_cell(r['Time_Overhead_mean'], r['Time_Overhead_ci95'])
                    
                    out.append(f"{prefix} & {deg_prefix} & {algo_lbl} & {cond} & {req} & {tot} & {oh} \\\\")
                out.append(r"\cmidrule(lr){2-7}")
            if g == 'er': out.append(r"\midrule")
        out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}" + "\n")

def gen_nonlinear(df_nl, out):
    """Experiment 2: Nonlinear Performance"""
    out.append(f"% --- Experiment 2: Nonlinear Causal Discovery ---")
    out.append(r"\begin{table*}[t!]\centering")
    out.append(r"\caption{Nonlinear Performance using KCI ($|\mathbf{V}|=10, M=1000$).}\label{table:nonlinear}")
    out.append(r"\begin{adjustbox}{width=\linewidth}\begin{tabular}{@{}llcccccccc@{}}\toprule")
    out.append(r"Graph & Method & F1 & Precision & Recall & SHD & Act. CIT & Cond. CIT & Tot. Time \\ \midrule")
    for g in ['er', 'sf']:
        sub_g = df_nl[df_nl['Graph'] == g]
        best_f1 = sub_g['F1_mean'].max()
        best_cit = sub_g['Cond_CIT_mean'].min()
        for i, algo in enumerate(["Standard PC", "DF-PC (Pure)"]):
            row = sub_g[sub_g['Algo'] == algo].iloc[0]
            prefix = f"\\multirow{{2}}{{*}}{{{g.upper()}}}" if i == 0 else ""
            algo_lbl = "Standard PC" if algo == "Standard PC" else r"\textbf{DF-PC}"
            f1 = format_cell(row['F1_mean'], row['F1_ci95'], row['F1_mean'] >= best_f1 - 1e-4)
            pr = format_cell(row['Prec_mean'], row['Prec_ci95'])
            re = format_cell(row['Rec_mean'], row['Rec_ci95'])
            sh = format_cell(row['SHD_mean'], row['SHD_ci95'], lower_is_better=True)
            act = format_cell(row['Actual_CIT_mean'], row['Actual_CIT_ci95'])
            cit = format_cell(row['Cond_CIT_mean'], row['Cond_CIT_ci95'], row['Cond_CIT_mean'] <= best_cit + 1e-4, True)
            tm = format_cell(row['Time_Total_mean'], row['Time_Total_ci95'])
            out.append(f"{prefix} & {algo_lbl} & {f1} & {pr} & {re} & {sh} & {act} & {cit} & {tm} \\\\")
        if g == 'er': out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table*}" + "\n")

def gen_extreme_cases(df_ex, out):
    """Experiment 3: Extreme Cases"""
    out.append(f"% --- Experiment 3: Extreme Case Hubs ---")
    out.append(r"\begin{table}[h!]\centering")
    out.append(r"\caption{Performance on Extreme Causal Hubs ($|\mathbf{V}|=11, M=2000$).}\label{table:extreme}")
    out.append(r"\begin{adjustbox}{width=\columnwidth}\begin{tabular}{@{}llcccc@{}}\toprule")
    out.append(r"Structure & Method & F1 & Total CIT & Time (s) \\ \midrule")
    for case in ['Collider-Hub', 'Source-Hub']:
        sub_c = df_ex[df_ex['Case'] == case]
        best_cit = sub_c['CIT'].min()
        for i, algo in enumerate(["Standard PC", "DF-PC (Pure)"]):
            row = sub_c[sub_c['Algo'] == algo].iloc[0]
            prefix = f"\\multirow{{2}}{{*}}{{{case}}}" if i == 0 else ""
            algo_lbl = "Standard PC" if algo == "Standard PC" else r"\textbf{DF-PC}"
            f1 = format_cell(row['F1'])
            cit = format_cell(row['CIT'], is_best=(row['CIT'] <= best_cit + 1e-4))
            tm = format_cell(row['Time'])
            out.append(f"{prefix} & {algo_lbl} & {f1} & {cit} & {tm} \\\\")
        if case == 'Collider-Hub': out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}" + "\n")

def gen_alpha_sensitivity(df_al, out):
    """Experiment 4: Alpha Sensitivity"""
    out.append(f"% --- Experiment 4: Alpha Sensitivity ---")
    out.append(r"\begin{table}[h!]\centering")
    out.append(r"\caption{Sensitivity Analysis of $\alpha$ ($N=40, M=5000$).}\label{table:alpha-sens}")
    out.append(r"\begin{adjustbox}{width=\columnwidth}\begin{tabular}{@{}llcccc@{}}\toprule")
    out.append(r"$\alpha$ & Method & F1 & Total CIT & Time (s) \\ \midrule")
    alphas = sorted(df_al['Alpha'].unique(), reverse=True)
    for a in alphas:
        sub_a = df_al[df_al['Alpha'] == a]
        best_cit = sub_a['CIT'].min()
        best_f1 = sub_a['F1'].max()
        for i, algo in enumerate(["Standard PC", "DF-PC (Pure)"]):
            row = sub_a[sub_a['Algo'] == algo].iloc[0]
            prefix = f"\\multirow{{2}}{{*}}{{{a}}}" if i == 0 else ""
            algo_lbl = "Standard PC" if algo == "Standard PC" else r"\textbf{DF-PC}"
            f1 = format_cell(row['F1'], is_best=(row['F1'] >= best_f1 - 1e-4))
            cit = format_cell(row['CIT'], is_best=(row['CIT'] <= best_cit + 1e-4))
            tm = format_cell(row['Time'])
            out.append(f"{prefix} & {algo_lbl} & {f1} & {cit} & {tm} \\\\")
        if a != alphas[-1]: out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}" + "\n")

def gen_scalability(df_sc, out):
    """Experiment 5: Scalability (N=100)"""
    out.append(f"% --- Experiment 5: Scalability Benchmarks (N=100) ---")
    out.append(r"\begin{table*}[t!]\centering")
    out.append(r"\caption{Scalability Results for Large Networks ($|\mathbf{V}|=100, M=1000$).}\label{table:scale}")
    out.append(r"\begin{adjustbox}{width=\linewidth}\begin{tabular}{@{} c c l c c c c c c c c @{}}\toprule")
    out.append(r"Graph & $D$ & Method & F1 & Precision & Recall & SHD & Act. CIT & Cond. CIT & Req. & Time (s) \\ \midrule")
    for g in ['er', 'sf']:
        for d in [2, 4]:
            sub = df_sc[(df_sc['Graph']==g) & (df_sc['Deg']==d)]
            best_f1 = sub['F1_mean'].max()
            best_cit = sub['Cond_CIT_mean'].min()
            for i, algo in enumerate(["Standard PC", "DF-PC (Pure)"]):
                row = sub[sub['Algo'] == algo].iloc[0]
                prefix = f"\\multirow{{2}}{{*}}{{{g.upper()}}}" if i == 0 and d == 2 else ""
                deg_prefix = f"\\multirow{{2}}{{*}}{{{d}}}" if i == 0 else ""
                algo_lbl = "Standard PC" if algo == "Standard PC" else r"\textbf{DF-PC}"
                f1 = format_cell(row['F1_mean'], row['F1_ci95'], row['F1_mean'] >= best_f1 - 1e-4)
                pr = format_cell(row['Prec_mean'], row['Prec_ci95'])
                re = format_cell(row['Rec_mean'], row['Rec_ci95'])
                sh = format_cell(row['SHD_mean'], row['SHD_ci95'], lower_is_better=True)
                act = format_cell(row['Actual_CIT_mean'], row['Actual_CIT_ci95'])
                cit = format_cell(row['Cond_CIT_mean'], row['Cond_CIT_ci95'], row['Cond_CIT_mean'] <= best_cit + 1e-4, True)
                req = format_cell(row['Requests_mean'], row['Requests_ci95'])
                tm = format_cell(row['Time_Total_mean'], row['Time_Total_ci95'])
                out.append(f"{prefix} & {deg_prefix} & {algo_lbl} & {f1} & {pr} & {re} & {sh} & {act} & {cit} & {req} & {tm} \\\\")
            out.append(r"\cmidrule(lr){2-11}")
        if g == 'er': out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table*}" + "\n")

def gen_realworld(df_rw, out):
    """Experiment 6: Real-World Benchmarks"""
    out.append(f"% --- Experiment 6: Real-World Network Benchmarks ---")
    out.append(r"\begin{table*}[t!]\centering")
    out.append(r"\caption{Real-World Benchmarks (Barley, Mildew).}\label{table:realworld}")
    out.append(r"\begin{adjustbox}{width=\linewidth}\begin{tabular}{@{} c c l c c c c c c c c @{}}\toprule")
    out.append(r"Dataset & $M$ & Method & F1 & Precision & Recall & SHD & Act. CIT & Cond. CIT & Req. & Time (s) \\ \midrule")
    for ds in ['barley', 'mildew']:
        for m in [2000, 5000]:
            sub = df_rw[(df_rw['Dataset']==ds) & (df_rw['Samples']==m)]
            best_f1 = sub['F1_mean'].max()
            best_cit = sub['Cond_CIT_mean'].min()
            for i, algo in enumerate(["Standard PC", "DF-PC (Pure)"]):
                row = sub[sub['Algo'] == algo].iloc[0]
                prefix = f"\\multirow{{4}}{{*}}{{{ds.capitalize()}}}" if i == 0 and m == 2000 else ""
                m_prefix = f"\\multirow{{2}}{{*}}{{{m}}}" if i == 0 else ""
                algo_lbl = "Standard PC" if algo == "Standard PC" else r"\textbf{DF-PC}"
                f1 = format_cell(row['F1_mean'], row['F1_ci95'], row['F1_mean'] >= best_f1 - 1e-4)
                pr = format_cell(row['Prec_mean'], row['Prec_ci95'])
                re = format_cell(row['Rec_mean'], row['Rec_ci95'])
                sh = format_cell(row['SHD_mean'], row['SHD_ci95'], lower_is_better=True)
                act = format_cell(row['Actual_CIT_mean'], row['Actual_CIT_ci95'])
                cit = format_cell(row['Cond_CIT_mean'], row['Cond_CIT_ci95'], row['Cond_CIT_mean'] <= best_cit + 1e-4, True)
                req = format_cell(row['Requests_mean'], row['Requests_ci95'])
                tm = format_cell(row['Time_Total_mean'], row['Time_Total_ci95'])
                out.append(f"{prefix} & {m_prefix} & {algo_lbl} & {f1} & {pr} & {re} & {sh} & {act} & {cit} & {req} & {tm} \\\\")
            out.append(r"\cmidrule(lr){2-11}")
        if ds == 'barley': out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table*}" + "\n")

def main():
    res_dir = Path("results")
    out_file = res_dir / "final_paper_tables.tex"
    out = [r"% Generated LaTeX Tables for DF-PC Paper", r"\newcommand{\stdv}[1]{\scriptsize{$\pm$#1}}", ""]
    
    # 1. Synthetic
    try:
        df_syn = pd.read_csv(res_dir / "synthetic_benchmarks_summary.csv")
        gen_synthetic_benchmarks(df_syn, out)
        gen_synthetic_efficiency(df_syn, out)
    except Exception as e: out.append(f"% Error Experiment 1: {e}")
    
    # 2. Nonlinear
    try: gen_nonlinear(pd.read_csv(res_dir / "experiment_nonlinear_summary.csv"), out)
    except Exception as e: out.append(f"% Error Experiment 2: {e}")
    
    # 3. Extreme
    try: gen_extreme_cases(pd.read_csv(res_dir / "experiment_extreme_cases.csv"), out)
    except Exception as e: out.append(f"% Error Experiment 3: {e}")
    
    # 4. Alpha
    try: gen_alpha_sensitivity(pd.read_csv(res_dir / "experiment_alpha_sensitivity.csv"), out)
    except Exception as e: out.append(f"% Error Experiment 4: {e}")
    
    # 5. Scalability
    try: gen_scalability(pd.read_csv(res_dir / "experiment_scalability_summary.csv"), out)
    except Exception as e: out.append(f"% Error Experiment 5: {e}")
    
    # 6. Real-World
    try: gen_realworld(pd.read_csv(res_dir / "experiment_realworld_summary.csv"), out)
    except Exception as e: out.append(f"% Error Experiment 6: {e}")
    
    with open(out_file, "w") as f:
        f.write("\n".join(out))
    print(f"Success! All tables consolidated into {out_file}")

if __name__ == "__main__":
    main()
