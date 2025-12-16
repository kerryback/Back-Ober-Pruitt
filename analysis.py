"""
Analysis script to produce LaTeX tables and figures from NoIPCA results.

This script replicates the analysis from analysis.ipynb, adapted to work with
the NoIPCA pickle file structure.

Usage:
    python analysis.py

Outputs:
    - Figures saved to NoIPCA/figures/
    - LaTeX tables saved to NoIPCA/tables/
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR

# Output directories
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR / "figures"
TABLES_DIR = SCRIPT_DIR / "tables"

# Create output directories if they don't exist
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def load_panel_data(data_dir: str, panel_id: str) -> pd.DataFrame:
    """Load the panel data (excess returns, characteristics, etc.)."""
    arrays_file = os.path.join(data_dir, f"{panel_id}_arrays.pkl")
    if os.path.exists(arrays_file):
        with open(arrays_file, 'rb') as f:
            data = pickle.load(f)
            return data['panel']
    return None


def compute_monthly_stats(returns_df: pd.DataFrame, portfolio_weights: np.ndarray,
                         panel_data: pd.DataFrame, start: int, end: int) -> List[Dict]:
    """
    Compute monthly statistics for factor returns.

    Args:
        returns_df: DataFrame of factor returns (T x K)
        portfolio_weights: Portfolio weights for SDF (optional, can be None)
        panel_data: Panel data with excess returns
        start: Start month
        end: End month

    Returns:
        List of dictionaries with monthly statistics
    """
    stats = []

    for month in range(start, end + 1):
        # Get excess returns for this month
        month_data = panel_data[panel_data['month'] == month]
        if len(month_data) == 0:
            continue

        xret = month_data['xret'].values

        # For Fama methods, we don't have explicit SDF portfolios in the same way
        # We'll need to compute this differently
        stats.append({
            'month': month,
            'xret': xret.mean(),
            'mn': returns_df.iloc[month - start].mean() if month - start < len(returns_df) else np.nan,
            'stdev': returns_df.iloc[month - start].std() if month - start < len(returns_df) else np.nan,
        })

    return stats


def load_fama_results(data_dir: str, model: str, n_panels: int) -> pd.DataFrame:
    """
    Load Fama-French and Fama-MacBeth results from pickle files.

    Returns DataFrame with columns: panel, method, alpha, month, mean, stdev, xret, hjd
    """
    print("  Loading Fama results...")
    all_results = []

    for i in range(n_panels):
        panel_id = f"{model}_{i}"
        fama_file = os.path.join(data_dir, f"{panel_id}_fama.pkl")

        if not os.path.exists(fama_file):
            continue

        with open(fama_file, 'rb') as f:
            fama_data = pickle.load(f)

        # Get the stats dataframe which has monthly data
        fama_stats = fama_data.get('fama_stats')
        if fama_stats is not None:
            # Add panel identifier
            fama_stats = fama_stats.copy()
            fama_stats['panel'] = i
            all_results.append(fama_stats)

    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        print(f"    Loaded {len(df)} Fama observations from {df['panel'].nunique()} panels")
        return df
    else:
        print("    No Fama results found")
        return pd.DataFrame()


def load_dkkm_results(data_dir: str, model: str, n_panels: int,
                     nfeatures_list: List[int] = [6, 36, 360]) -> pd.DataFrame:
    """
    Load DKKM results from pickle files.

    Returns DataFrame with columns: panel, factors, kappa, sharpe, max_sr
    """
    print("  Loading DKKM results...")
    all_results = []

    for i in range(n_panels):
        panel_id = f"{model}_{i}"

        for nfeatures in nfeatures_list:
            dkkm_file = os.path.join(data_dir, f"{panel_id}_dkkm_{nfeatures}.pkl")

            if not os.path.exists(dkkm_file):
                continue

            with open(dkkm_file, 'rb') as f:
                dkkm_data = pickle.load(f)

            # Get the stats dataframe which has monthly data
            dkkm_stats = dkkm_data.get('dkkm_stats')
            if dkkm_stats is not None:
                # Add panel identifier and nfeatures
                dkkm_stats = dkkm_stats.copy()
                dkkm_stats['panel'] = i
                dkkm_stats['factors'] = nfeatures
                all_results.append(dkkm_stats)

    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        print(f"    Loaded {len(df)} DKKM observations from {df['panel'].nunique()} panels")
        return df
    else:
        print("    No DKKM results found")
        return pd.DataFrame()


def load_results(data_dir: str, model: str = 'bgn', n_panels: int = 300,
                nfeatures_list: List[int] = [6, 36, 360]) -> Dict:
    """
    Load all results from pickle files.

    Args:
        data_dir: Directory containing pickle files
        model: Model name (bgn, kp14, gs21)
        n_panels: Number of panel iterations to load
        nfeatures_list: List of DKKM feature counts to load

    Returns:
        Dictionary with 'fama' and 'dkkm' DataFrames
    """
    print(f"Loading results from {data_dir}...")
    print(f"  Model: {model}")
    print(f"  Number of panels: {n_panels}")
    print(f"  DKKM features: {nfeatures_list}")

    fama_df = load_fama_results(data_dir, model, n_panels)
    dkkm_df = load_dkkm_results(data_dir, model, n_panels, nfeatures_list)

    return {
        'fama': fama_df,
        'dkkm': dkkm_df
    }


def create_means_figure(dkkm: pd.DataFrame, kps: pd.DataFrame,
                       ffc: pd.DataFrame, fmr: pd.DataFrame,
                       output_path: str):
    """
    Create the combined means figure with 4 subplots.

    Args:
        dkkm: DKKM results DataFrame
        kps: KPS/IPCA results DataFrame
        ffc: Fama-French results DataFrame
        fmr: Fama-MacBeth results DataFrame
        output_path: Path to save figure
    """
    print("Creating means figure...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

    # DKKM Sharpe plot
    dkkm_filtered = dkkm[dkkm['kappa'] != 0].copy()
    mean_sharpe = dkkm_filtered.groupby(["kappa", "factors"]).sharpe.mean().reset_index()
    sns.lineplot(data=mean_sharpe, x='factors', y='sharpe', hue='kappa',
                marker='o', linewidth=2, ax=ax1)
    ax1.set_xlabel('DKKM Factors', fontsize=10)
    ax1.set_title('(a) DKKM Sharpe Ratio')
    ax1.set_ylabel('', fontsize=12)
    ax1.legend(title='Kappa', fontsize=10)
    ax1.set_xscale('log')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # DKKM HJD plot
    mean_hjd = dkkm_filtered.groupby(["kappa", "factors"]).hjd.mean().reset_index()
    sns.lineplot(data=mean_hjd, x='factors', y='hjd', hue='kappa',
                marker='o', linewidth=2, ax=ax2)
    ax2.set_xlabel('DKKM Factors', fontsize=10)
    ax2.set_title('(b) DKKM HJ Distance')
    ax2.set_ylabel('')
    ax2.legend(title='Kappa', fontsize=10)
    ax2.set_xscale('log')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # KPS Sharpe plot
    max_sharpe = dkkm[(dkkm.factors==6) & (dkkm.kappa==0.05)].max_sr.mean()
    mean_sharpe = kps.groupby(["factors"]).sharpe.mean().reset_index()
    sns.lineplot(data=mean_sharpe, x='factors', y='sharpe', marker='o',
                linewidth=1, ax=ax3, label="KPS")
    sns.lineplot(x=[1, 6], y=[fmr.sharpe.mean()]*2, ax=ax3, label="FMR")
    sns.lineplot(x=[1, 6], y=[ffc.sharpe.mean()]*2, ax=ax3, label="FFC")
    sns.lineplot(x=[1, 6], y=[max_sharpe]*2, ax=ax3, label="Max SR")

    ax3.set_xlabel('KPS Factors', fontsize=10)
    ax3.set_title('(c) Sharpe Ratio')
    ax3.set_ylabel('', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    # KPS HJD plot
    mean_hjd = kps.groupby(["factors"]).hjd.mean().reset_index()
    sns.lineplot(data=mean_hjd, x='factors', y='hjd', marker='o',
                linewidth=1, ax=ax4, label="KPS")
    sns.lineplot(x=[1, 6], y=[fmr.hjd.mean()]*2, ax=ax4, label="FMR")
    sns.lineplot(x=[1, 6], y=[ffc.hjd.mean()]*2, ax=ax4, label="FFC")
    ax4.set_xlabel('KPS Factors', fontsize=10)
    ax4.set_title('(d) HJ Distance')
    ax4.set_ylabel('')
    ax4.legend()
    ax4.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"  Saved to {output_path}")
    plt.close()


def create_dkkm_sharpe_figure(dkkm: pd.DataFrame, output_path: str):
    """Create DKKM Sharpe ratios figure."""
    print("Creating DKKM Sharpe figure...")

    fig, ax1 = plt.subplots(figsize=(6, 4))

    dkkm_filtered = dkkm[dkkm['kappa'] != 0].copy()
    print(f"  Filtered DKKM data: {len(dkkm_filtered)} observations")

    if len(dkkm_filtered) == 0:
        print("  WARNING: No data after filtering kappa != 0")
        plt.close()
        return

    mean_sharpe = dkkm_filtered.groupby(["kappa", "factors"]).sharpe.mean().reset_index()
    print(f"  Mean sharpe data: {len(mean_sharpe)} observations")
    print(f"  Columns: {list(mean_sharpe.columns)}")
    print(f"  Data types: {mean_sharpe.dtypes.to_dict()}")
    print(f"  First few rows:")
    print(mean_sharpe.head())

    if len(mean_sharpe) == 0:
        print("  WARNING: No data after groupby")
        plt.close()
        return

    # Check for NaN/Inf values
    if mean_sharpe['sharpe'].isna().any():
        print(f"  WARNING: {mean_sharpe['sharpe'].isna().sum()} NaN values in sharpe")
        mean_sharpe = mean_sharpe.dropna()

    if len(mean_sharpe) == 0:
        print("  WARNING: No data after dropping NaN")
        plt.close()
        return

    sns.lineplot(data=mean_sharpe, x='factors', y='sharpe', hue='kappa',
                marker='o', linewidth=2, ax=ax1)
    ax1.set_xlabel('Number of Factors', fontsize=14)
    ax1.set_ylabel('', fontsize=12)
    ax1.legend(title='Penalization', fontsize=10)
    ax1.set_xscale('log')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"  Saved to {output_path}")
    plt.close()


def create_kps_sharpe_figure(dkkm: pd.DataFrame, kps: pd.DataFrame,
                             ffc: pd.DataFrame, fmr: pd.DataFrame,
                             output_path: str):
    """Create KPS Sharpe ratios figure."""
    print("Creating KPS Sharpe figure...")

    fig, ax3 = plt.subplots(figsize=(6, 4))

    max_sharpe = dkkm[(dkkm.factors==6) & (dkkm.kappa==0.05)].max_sr.mean()
    mean_sharpe = kps.groupby(["factors"]).sharpe.mean().reset_index()
    sns.lineplot(data=mean_sharpe, x='factors', y='sharpe', marker='o',
                linewidth=1, ax=ax3, label="KPS")
    sns.lineplot(x=[1, 6], y=[fmr.sharpe.mean()]*2, ax=ax3, label="FMR")
    sns.lineplot(x=[1, 6], y=[ffc.sharpe.mean()]*2, ax=ax3, label="FFC")
    sns.lineplot(x=[1, 6], y=[max_sharpe]*2, ax=ax3, label="Max SR")

    ax3.set_xlabel('Number of IPCA Factors', fontsize=14)
    ax3.set_ylabel('', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"  Saved to {output_path}")
    plt.close()


def create_densities_figure(dkkm: pd.DataFrame, kps: pd.DataFrame,
                            fmr: pd.DataFrame, output_path: str):
    """Create densities figure with Sharpe and HJD distributions."""
    print("Creating densities figure...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    dkkm_subset = dkkm[(dkkm.kappa == 0.05) & (dkkm.factors == 360)]
    kps_subset = kps[kps.factors == 2]

    # Sharpe ratios
    sns.kdeplot(data=dkkm_subset.sharpe, fill=True, alpha=0.5, label='DKKM', ax=ax1)
    sns.kdeplot(data=kps_subset.sharpe, fill=True, alpha=0.5, label='KPS', ax=ax1)
    sns.kdeplot(data=fmr.sharpe, fill=True, alpha=0.5, label='FMR', ax=ax1)
    ax1.set_xlabel('Sharpe Ratio')
    ax1.set_ylabel('Density')
    ax1.legend()

    # HJD
    sns.kdeplot(data=dkkm_subset.hjd, fill=True, alpha=0.5, label='DKKM', ax=ax2)
    sns.kdeplot(data=kps_subset.hjd, fill=True, alpha=0.5, label='KPS', ax=ax2)
    sns.kdeplot(data=fmr.hjd, fill=True, alpha=0.5, label='FMR', ax=ax2)
    ax2.set_xlabel('Hansen-Jagannathan Distance')
    ax2.set_ylabel('')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"  Saved to {output_path}")
    plt.close()


def create_fama_table(ffc: pd.DataFrame, fmr: pd.DataFrame,
                     output_path: str, n_panels: int = 300):
    """Create Fama comparison table."""
    print("Creating Fama comparison table...")

    table = pd.DataFrame(
        index=["FMR", "FFC", "FMR - FFC", "t-stat", "p-value"],
        columns=["Sharpe Ratio", "HJ Distance"],
        dtype=float
    )

    ffc_hjd = ffc.groupby("panel").hjd.mean()
    fmr_hjd = fmr.groupby("panel").hjd.mean()

    table.loc["FMR", "Sharpe Ratio"] = fmr.sharpe.mean()
    table.loc["FMR", "HJ Distance"] = fmr_hjd.mean()
    table.loc["FFC", "Sharpe Ratio"] = ffc.sharpe.mean()
    table.loc["FFC", "HJ Distance"] = ffc_hjd.mean()
    table.loc["FMR - FFC", "Sharpe Ratio"] = fmr.sharpe.mean() - ffc.sharpe.mean()
    table.loc["FMR - FFC", "HJ Distance"] = fmr_hjd.mean() - ffc_hjd.mean()

    test = ttest_1samp(
        fmr.groupby("panel").sharpe.mean().to_numpy()
        - ffc.groupby("panel").sharpe.mean().to_numpy(),
        0
    )
    table.loc["t-stat", "Sharpe Ratio"] = test[0]
    table.loc["p-value", "Sharpe Ratio"] = test[1]

    test = ttest_1samp(
        fmr_hjd.to_numpy() - ffc_hjd.to_numpy(),
        0
    )
    table.loc["t-stat", "HJ Distance"] = test[0]
    table.loc["p-value", "HJ Distance"] = test[1]

    # Create LaTeX table
    latex_table = """\\begin{table}[h]
\\centering
"""
    latex_table += table.to_latex(float_format="%.3f")
    latex_table += f"""
\\caption{{
    \\textbf{{Performance of FMR and FFC Models.}}
    {n_panels} panels of data are generated for the BGN economy.  The conditional Sharpe
    ratio and $(\\hat y_{{t+1}}-z_{{t+1}})^2$ are calculated for each out-of-sample month for the FMR and FFC models, and
    the means are computed for each panel.  The $t$-statistics and $p$-values are
    for the difference between the FMR and FFC panel means.
    \\label{{tab:fama}}}}
\\end{{table}}"""

    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"  Saved to {output_path}")


def create_dkkm_table(dkkm: pd.DataFrame, fmr: pd.DataFrame,
                     output_path: str, n_panels: int = 300):
    """Create DKKM comparison table."""
    print("Creating DKKM comparison table...")

    t_stats_sharpe = pd.DataFrame(
        index=dkkm.kappa.unique(),
        columns=dkkm.factors.unique(),
        dtype=float
    )

    t_stats_hjd = pd.DataFrame(
        index=dkkm.kappa.unique(),
        columns=dkkm.factors.unique(),
        dtype=float
    )

    # Fill DataFrames with t-stats
    for kappa in t_stats_sharpe.index:
        for factors in t_stats_sharpe.columns:
            dkkm_sharpe = dkkm[(dkkm.kappa == kappa) & (dkkm.factors == factors)].sharpe.to_numpy()
            t_stat_sharpe, _ = ttest_1samp(dkkm_sharpe - fmr.sharpe.to_numpy(), 0)
            t_stats_sharpe.loc[kappa, factors] = t_stat_sharpe

            dkkm_hjd = dkkm[(dkkm.kappa == kappa) & (dkkm.factors == factors)].hjd.to_numpy()
            t_stat_hjd, _ = ttest_1samp(dkkm_hjd - fmr.hjd.to_numpy(), 0)
            t_stats_hjd.loc[kappa, factors] = t_stat_hjd

    # Create LaTeX table
    latex_table = """\\begin{table}[h]
\\centering
\\begin{subtable}{\\textwidth}
\\centering
\\caption{Sharpe Ratio}
"""
    latex_table += t_stats_sharpe.to_latex(float_format="%.2f")
    latex_table += """\\end{subtable}

\\begin{subtable}{\\textwidth}
\\centering
\\caption{Hansen-Jagannathan Distance}
"""
    latex_table += t_stats_hjd.to_latex(float_format="%.2f")
    latex_table += f"""\\end{{subtable}}
\\caption{{
    \\textbf{{Performance of the DKKM Model.}}
    {n_panels} panels of data are generated for the BGN economy.  The conditional Sharpe
    ratio and $(\\hat y_{{t+1}}-z_{{t+1}})^2$ are calculated for each out-of-sample month for the DKKM and FMR models, and
    the means are computed for each panel.  The table reports $t$-statistics
   for the difference between the DKKM and FMR panel means.
    \\label{{tab:dkkm}}}}
\\end{{table}}"""

    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"  Saved to {output_path}")


def create_kps_table(kps: pd.DataFrame, dkkm: pd.DataFrame, fmr: pd.DataFrame,
                    output_path: str, n_panels: int = 300):
    """Create KPS/IPCA comparison table."""
    print("Creating KPS comparison table...")

    t_stats_kps_dkkm = pd.DataFrame(
        index=["vs DKKM", "vs FMR"],
        columns=kps.factors.unique(),
        dtype=float
    )

    t_stats_kps_dkkm_hjd = pd.DataFrame(
        index=["vs DKKM", "vs FMR"],
        columns=kps.factors.unique(),
        dtype=float
    )

    # Fill DataFrames with t-stats
    dkkm_subset = dkkm[(dkkm.kappa == 0.05) & (dkkm.factors == 360)].sharpe.to_numpy()
    fm_sharpe = fmr.sharpe.to_numpy()

    for factors in t_stats_kps_dkkm.columns:
        kps_subset = kps[kps.factors == factors].sharpe.to_numpy()

        t_stat, _ = ttest_1samp(kps_subset - dkkm_subset, 0)
        t_stats_kps_dkkm.loc["vs DKKM", factors] = t_stat

        t_stat, _ = ttest_1samp(kps_subset - fm_sharpe, 0)
        t_stats_kps_dkkm.loc["vs FMR", factors] = t_stat

    # HJD t-stats
    dkkm_subset = dkkm[(dkkm.kappa == 0.05) & (dkkm.factors == 360)].hjd.to_numpy()
    fm_hjd = fmr.hjd.to_numpy()

    for factors in t_stats_kps_dkkm_hjd.columns:
        kps_subset = kps[kps.factors == factors].hjd.to_numpy()

        t_stat, _ = ttest_1samp(kps_subset - dkkm_subset, 0)
        t_stats_kps_dkkm_hjd.loc["vs DKKM", factors] = t_stat

        t_stat, _ = ttest_1samp(kps_subset - fm_hjd, 0)
        t_stats_kps_dkkm_hjd.loc["vs FMR", factors] = t_stat

    # Create LaTeX table
    latex_table = """\\begin{table}[h]
\\centering
\\begin{subtable}{\\textwidth}
\\centering
\\caption{Sharpe Ratio}
"""
    latex_table += t_stats_kps_dkkm.to_latex(float_format="%.2f")
    latex_table += """\\end{subtable}

\\begin{subtable}{\\textwidth}
\\centering
\\caption{Hansen-Jagannathan Distance}
"""
    latex_table += t_stats_kps_dkkm_hjd.to_latex(float_format="%.2f")
    latex_table += f"""\\end{{subtable}}
\\caption{{
    \\textbf{{Performance of the KPS Model.}}
    {n_panels} panels of data are generated for the BGN economy.  The
    conditional Sharpe
    ratio and $(\\hat y_{{t+1}}-z_{{t+1}})^2$ are calculated for each out-of-sample month for the DKKM, FMR, and KPS models, and
    the means are computed for each panel.  The table reports $t$-statistics
   for the difference in panel means of the KPS model compared to the DKKM model with 360 factors
   and $\\kappa=0.05$ and the
    KPS model compared to the FMR model.
    \\label{{tab:kps}}}}
\\end{{table}}"""

    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"  Saved to {output_path}")


def main():
    """Main analysis function."""
    print("="*70)
    print("NoIPCA ANALYSIS")
    print("="*70)
    print()

    print(f"Output directories:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Data directory: {DATA_DIR}")
    print()

    # Configuration
    model = 'bgn'
    n_panels = 1  # Start with 1 panel for testing, increase to 300 for full analysis
    nfeatures_list = [6, 36, 360]

    # Load results
    print("Loading data...")
    results = load_results(DATA_DIR, model=model, n_panels=n_panels,
                          nfeatures_list=nfeatures_list)

    fama = results['fama']
    dkkm = results['dkkm']

    if len(fama) == 0:
        print("ERROR: No Fama results found. Make sure you have run the workflow first:")
        print(f"  python main.py {model} 0")
        return

    if len(dkkm) == 0:
        print("ERROR: No DKKM results found. Make sure you have run the workflow first:")
        print(f"  python main.py {model} 0")
        return

    print()
    print("Processing data...")

    # Rename columns to match original notebook
    fama = fama.rename(columns={"method": "model", "mean": "mn"})
    dkkm = dkkm.rename(columns={"alpha": "kappa", "mean": "mn"})

    # Calculate Sharpe ratios
    fama["sharpe"] = fama.mn / fama.stdev
    dkkm["sharpe"] = dkkm.mn / dkkm.stdev

    # Calculate HJD realized (squared difference)
    # Note: hjd column should already exist in the data but may be NaN
    # For now we'll use the existing column or set to 0
    if 'hjd' not in fama.columns:
        fama["hjd_realized"] = 0.0
        fama["hjd"] = 0.0
    else:
        fama["hjd_realized"] = fama.hjd ** 2

    if 'hjd' not in dkkm.columns:
        dkkm["hjd_realized"] = 0.0
        dkkm["hjd"] = 0.0
    else:
        dkkm["hjd_realized"] = dkkm.hjd ** 2

    # Aggregate monthly data to panel-level statistics
    # This matches the original notebook's approach
    print("  Aggregating monthly data to panel level...")
    fama = fama.groupby(["model", "panel", "alpha"])[["sharpe", "hjd_realized"]].mean().reset_index()
    dkkm = dkkm.groupby(["kappa", "factors", "panel"])[["sharpe", "hjd_realized"]].mean().reset_index()

    # Calculate HJD from realized values
    fama["hjd"] = np.sqrt(fama.hjd_realized)
    dkkm["hjd"] = np.sqrt(dkkm.hjd_realized)

    # Filter to alpha=0 for Fama methods (no shrinkage)
    fama = fama[fama.alpha == 0].copy()
    fama = fama.drop(columns=["alpha"])

    # Split by method
    ffc = fama[fama.model == "ff"].copy()
    fmr = fama[fama.model == "fm"].copy()

    print(f"  Fama-French results: {len(ffc)} panel observations")
    print(f"  Fama-MacBeth results: {len(fmr)} panel observations")
    print(f"  DKKM results: {len(dkkm)} panel observations")
    print()

    # For now, skip KPS analysis (not implemented yet)
    # Create a dummy KPS dataframe for compatibility
    kps = pd.DataFrame({
        'factors': [1, 2, 3, 4, 5, 6],
        'sharpe': [0.5, 0.6, 0.7, 0.75, 0.78, 0.8],
        'hjd': [0.3, 0.25, 0.2, 0.18, 0.17, 0.16],
        'panel': [0] * 6
    })

    try:
        # Create figures only if we have enough data
        if len(ffc) > 0 and len(fmr) > 0:
            print("Creating figures...")

            if len(dkkm) > 0:
                print("  Creating DKKM Sharpe figure...")
                create_dkkm_sharpe_figure(dkkm, FIGURES_DIR / "dkkm_sharpe.pdf")

            # Commented out until we have proper data
            # create_means_figure(dkkm, kps, ffc, fmr, FIGURES_DIR / "means.pdf")
            # create_kps_sharpe_figure(dkkm, kps, ffc, fmr, FIGURES_DIR / "kps_sharpe.pdf")
            # create_densities_figure(dkkm, kps, fmr, FIGURES_DIR / "densities.pdf")

        # Create tables only if we have enough data
        if len(ffc) > 0 and len(fmr) > 0:
            print("Creating tables...")
            print("  Creating Fama comparison table...")
            create_fama_table(ffc, fmr, TABLES_DIR / "fama_comparisons.tex", n_panels=n_panels)

            if len(dkkm) > 0:
                print("  Creating DKKM comparison table...")
                create_dkkm_table(dkkm, fmr, TABLES_DIR / "dkkm_comparisons.tex", n_panels=n_panels)

            # Commented out until we have KPS implementation
            # create_kps_table(kps, dkkm, fmr, TABLES_DIR / "kps_comparisons.tex", n_panels=n_panels)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print("Note: To run full analysis with 300 panels, change n_panels=300 in the script")
    print("Note: KPS/IPCA analysis not yet implemented - requires separate IPCA runs")


if __name__ == "__main__":
    main()
