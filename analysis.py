"""
Analysis script to produce LaTeX tables and figures from Back-Ober-Pruitt results.

This script automatically discovers and analyzes all panels for each model (BGN, KP14, GS21).

Usage:
    python analysis.py

Outputs:
    - Figures saved to Back-Ober-Pruitt/figures/
    - LaTeX tables saved to Back-Ober-Pruitt/tables/
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Configuration
MODELS = ['bgn', 'kp14', 'gs21']
DKKM_NFEATURES = [6, 36, 360]
IPCA_K_VALUES = [1, 2, 3]


def discover_panels(model: str) -> List[int]:
    """
    Discover all available panel indices for a given model by scanning pickle files.

    Args:
        model: Model name ('bgn', 'kp14', 'gs21')

    Returns:
        Sorted list of panel indices
    """
    panels = set()

    # Scan for fama files (most likely to exist)
    pattern = os.path.join(DATA_DIR, f"{model}_*_fama.pkl")
    for filepath in glob.glob(pattern):
        # Extract index from filename: model_index_fama.pkl
        filename = Path(filepath).stem
        parts = filename.split('_')
        if len(parts) >= 3 and parts[1].isdigit():
            panels.add(int(parts[1]))

    return sorted(list(panels))


def load_fama_results(model: str, panel_indices: List[int]) -> pd.DataFrame:
    """Load Fama results for all panels of a model."""
    all_results = []

    for panel_idx in panel_indices:
        panel_id = f"{model}_{panel_idx}"
        fama_file = os.path.join(DATA_DIR, f"{panel_id}_fama.pkl")

        if not os.path.exists(fama_file):
            continue

        with open(fama_file, 'rb') as f:
            fama_data = pickle.load(f)

        # Get the stats dataframe
        fama_stats = fama_data.get('fama_stats')
        if fama_stats is not None and len(fama_stats) > 0:
            fama_stats = fama_stats.copy()
            fama_stats['panel'] = panel_idx
            fama_stats['model_type'] = model
            all_results.append(fama_stats)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def load_dkkm_results(model: str, panel_indices: List[int],
                      nfeatures_list: List[int] = DKKM_NFEATURES) -> pd.DataFrame:
    """Load DKKM results for all panels of a model."""
    all_results = []

    for panel_idx in panel_indices:
        panel_id = f"{model}_{panel_idx}"

        for nfeatures in nfeatures_list:
            dkkm_file = os.path.join(DATA_DIR, f"{panel_id}_dkkm_{nfeatures}.pkl")

            if not os.path.exists(dkkm_file):
                continue

            with open(dkkm_file, 'rb') as f:
                dkkm_data = pickle.load(f)

            # Get the stats dataframe
            dkkm_stats = dkkm_data.get('dkkm_stats')
            if dkkm_stats is not None and len(dkkm_stats) > 0:
                dkkm_stats = dkkm_stats.copy()
                dkkm_stats['panel'] = panel_idx
                dkkm_stats['model_type'] = model
                dkkm_stats['num_factors'] = nfeatures
                all_results.append(dkkm_stats)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def load_ipca_results(model: str, panel_indices: List[int],
                      K_list: List[int] = IPCA_K_VALUES) -> pd.DataFrame:
    """Load IPCA results for all panels of a model."""
    all_results = []

    for panel_idx in panel_indices:
        panel_id = f"{model}_{panel_idx}"

        for K in K_list:
            ipca_file = os.path.join(DATA_DIR, f"{panel_id}_ipca_{K}.pkl")

            if not os.path.exists(ipca_file):
                continue

            with open(ipca_file, 'rb') as f:
                ipca_data = pickle.load(f)

            # Get the stats dataframe
            ipca_stats = ipca_data.get('ipca_stats')
            if ipca_stats is not None and len(ipca_stats) > 0:
                ipca_stats = ipca_stats.copy()
                ipca_stats['panel'] = panel_idx
                ipca_stats['model_type'] = model
                ipca_stats['num_factors'] = K
                all_results.append(ipca_stats)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def create_dkkm_figure(dkkm: pd.DataFrame, ffc: pd.DataFrame, fmr: pd.DataFrame,
                       output_path: str, model: str = 'bgn'):
    """Create DKKM figure with Sharpe and HJD panels."""
    print(f"  Creating DKKM figure for {model.upper()}...")

    # Filter out kappa = 0 only
    dkkm_filtered = dkkm[dkkm['kappa'] > 0].copy()

    if len(dkkm_filtered) == 0:
        print(f"    WARNING: No data after filtering for {model}")
        return

    # Compute group means across panels
    mean_sharpe = dkkm_filtered.groupby(['kappa', 'num_factors'])['sharpe'].mean().reset_index()
    mean_hjd = dkkm_filtered.groupby(['kappa', 'num_factors'])['hjd'].mean().reset_index()

    # Create side-by-side panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Sharpe ratios
    for kappa in mean_sharpe['kappa'].unique():
        data = mean_sharpe[mean_sharpe['kappa'] == kappa]
        ax1.plot(data['num_factors'], data['sharpe'], marker='o', linewidth=2, label=f'$\\kappa$={kappa:.4g}')

    ax1.axhline(ffc.sharpe.mean(), linewidth=2, label='FFC', color='lightgreen', linestyle='--')
    ax1.axhline(fmr.sharpe.mean(), linewidth=2, label='FMR', color='brown', linestyle='--')
    ax1.set_xlabel('Number of DKKM Factors', fontsize=12)
    ax1.set_ylabel('Mean Sharpe Ratio', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_title(f'(a) Sharpe Ratios - {model.upper()}')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    # Right panel: HJ distances
    for kappa in mean_hjd['kappa'].unique():
        data = mean_hjd[mean_hjd['kappa'] == kappa]
        ax2.plot(data['num_factors'], data['hjd'], marker='o', linewidth=2, label=f'$\\kappa$={kappa:.4g}')

    ax2.axhline(ffc.hjd.mean(), linewidth=2, label='FFC', color='lightgreen', linestyle='--')
    ax2.axhline(fmr.hjd.mean(), linewidth=2, label='FMR', color='brown', linestyle='--')
    ax2.set_xlabel('Number of DKKM Factors', fontsize=12)
    ax2.set_ylabel('Hansen-Jagannathan Distance', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_title(f'(b) HJ Distances - {model.upper()}')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"    Saved to {output_path}")
    plt.close()


def create_fama_table(fama_all: pd.DataFrame, output_path: str):
    """Create Fama comparison table for all models."""
    print("  Creating Fama comparison table...")

    latex_table = r"\begin{table}[h]" + "\n"
    latex_table += r"\centering" + "\n"
    latex_table += r"\begin{tabular}{llrr}" + "\n"
    latex_table += r"\toprule" + "\n"
    latex_table += r"Model & Method & Sharpe Ratio & HJ Distance \\" + "\n"
    latex_table += r"\midrule" + "\n"

    for model in MODELS:
        model_data = fama_all[fama_all['model_type'] == model]
        if len(model_data) == 0:
            continue

        # Get number of panels for this model
        n_panels = model_data['panel'].nunique()

        # Aggregate by method
        for method, label in [('fm', 'FMR'), ('ff', 'FFC')]:
            method_data = model_data[model_data['method'] == method]
            if len(method_data) > 0:
                sharpe = method_data['sharpe'].mean()
                hjd = method_data['hjd'].mean()
                latex_table += f"{model.upper()} & {label} & {sharpe:.4f} & {hjd:.4f} \\\\\n"

        # Add difference row
        fmr_data = model_data[model_data['method'] == 'fm']
        ffc_data = model_data[model_data['method'] == 'ff']
        if len(fmr_data) > 0 and len(ffc_data) > 0:
            sharpe_diff = fmr_data['sharpe'].mean() - ffc_data['sharpe'].mean()
            hjd_diff = fmr_data['hjd'].mean() - ffc_data['hjd'].mean()
            latex_table += f"{model.upper()} & FMR - FFC & {sharpe_diff:.4f} & {hjd_diff:.4f} \\\\\n"

        latex_table += r"\midrule" + "\n"

    latex_table += r"\bottomrule" + "\n"
    latex_table += r"\end{tabular}" + "\n"
    latex_table += "\n"
    latex_table += r"\caption{" + "\n"
    latex_table += r"    \textbf{Performance of FMR and FFC Models.}" + "\n"
    latex_table += r"    The conditional Sharpe ratio and Hansen-Jagannathan distance" + "\n"
    latex_table += r"    are calculated for each out-of-sample month and averaged across panels." + "\n"
    latex_table += r"    \label{tab:fama}}" + "\n"
    latex_table += r"\end{table}" + "\n"

    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"    Saved to {output_path}")


def create_dkkm_table(dkkm_all: pd.DataFrame, fama_all: pd.DataFrame, output_path: str):
    """Create DKKM comparison table for all models."""
    print("  Creating DKKM comparison table...")

    latex_table = r"\begin{table}[h]" + "\n"
    latex_table += r"\centering" + "\n"

    for metric, metric_name in [('sharpe', 'Sharpe Ratio'), ('hjd', 'Hansen-Jagannathan Distance')]:
        latex_table += r"\begin{subtable}{\textwidth}" + "\n"
        latex_table += r"\centering" + "\n"
        latex_table += f"\\caption{{{metric_name}}}\n"
        latex_table += r"\begin{tabular}{llrrr}" + "\n"
        latex_table += r"\toprule" + "\n"
        latex_table += r"Model & $\kappa$ & 6 & 36 & 360 \\" + "\n"
        latex_table += r"\midrule" + "\n"

        for model in MODELS:
            model_dkkm = dkkm_all[dkkm_all['model_type'] == model]
            model_fama = fama_all[(fama_all['model_type'] == model) & (fama_all['method'] == 'fm')]

            if len(model_dkkm) == 0 or len(model_fama) == 0:
                continue

            # Get unique kappa values for this model
            kappa_values = sorted(model_dkkm['kappa'].unique())

            for kappa in kappa_values:
                kappa_data = model_dkkm[model_dkkm['kappa'] == kappa]
                row = [model.upper(), f"{kappa:.6f}"]

                for nfeatures in DKKM_NFEATURES:
                    nf_data = kappa_data[kappa_data['num_factors'] == nfeatures]
                    if len(nf_data) > 0:
                        # Mean raw value
                        dkkm_value = nf_data[metric].mean()
                        row.append(f"{dkkm_value:.4f}")
                    else:
                        row.append("--")

                latex_table += " & ".join(row) + " \\\\\n"

            latex_table += r"\midrule" + "\n"

        latex_table += r"\bottomrule" + "\n"
        latex_table += r"\end{tabular}" + "\n"
        latex_table += r"\end{subtable}" + "\n"
        latex_table += "\n"

    latex_table += r"\caption{" + "\n"
    latex_table += r"    \textbf{Performance of the DKKM Model.}" + "\n"
    latex_table += r"    Values show raw Sharpe ratios and HJ distances." + "\n"
    latex_table += r"    Results averaged across panels." + "\n"
    latex_table += r"    \label{tab:dkkm}}" + "\n"
    latex_table += r"\end{table}" + "\n"

    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"    Saved to {output_path}")


def create_fama_boxplots(fama_all: pd.DataFrame, output_path: str):
    """Create boxplots showing distribution of Fama metrics across panels."""
    print("  Creating Fama boxplots...")

    # Calculate panel-level means for each model-method combination
    panel_means = fama_all.groupby(['model_type', 'method', 'panel']).agg({
        'sharpe': 'mean',
        'hjd': 'mean'
    }).reset_index()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Prepare data for boxplots
    boxplot_data_sharpe = []
    boxplot_data_hjd = []
    labels = []

    for model in MODELS:
        model_data = panel_means[panel_means['model_type'] == model]
        if len(model_data) == 0:
            continue

        for method, label in [('ff', 'FFC'), ('fm', 'FMR')]:
            method_data = model_data[model_data['method'] == method]
            if len(method_data) > 0:
                boxplot_data_sharpe.append(method_data['sharpe'].values)
                boxplot_data_hjd.append(method_data['hjd'].values)
                labels.append(f"{model.upper()}\n{label}")

    # Left panel: Sharpe ratios
    bp1 = ax1.boxplot(boxplot_data_sharpe, labels=labels, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.set_title('(a) Sharpe Ratios - Distribution Across Panels')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Right panel: HJ distances
    bp2 = ax2.boxplot(boxplot_data_hjd, labels=labels, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    ax2.set_ylabel('Hansen-Jagannathan Distance', fontsize=12)
    ax2.set_title('(b) HJ Distances - Distribution Across Panels')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"    Saved to {output_path}")
    plt.close()


def create_dkkm_boxplots(dkkm_all: pd.DataFrame, output_path: str, model: str = 'bgn'):
    """Create boxplots showing distribution of DKKM metrics across panels for a specific model."""
    print(f"  Creating DKKM boxplots for {model.upper()}...")

    # Filter for this model and kappa > 0
    model_dkkm = dkkm_all[(dkkm_all['model_type'] == model) & (dkkm_all['kappa'] > 0)]

    if len(model_dkkm) == 0:
        print(f"    WARNING: No data for {model}")
        return

    # Calculate panel-level means
    panel_means = model_dkkm.groupby(['kappa', 'num_factors', 'panel']).agg({
        'sharpe': 'mean',
        'hjd': 'mean'
    }).reset_index()

    # Get unique kappa and factor values
    kappa_values = sorted(panel_means['kappa'].unique())

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data for boxplots - organize by num_factors, with kappa as hue
    for nfeatures in DKKM_NFEATURES:
        factor_data = panel_means[panel_means['num_factors'] == nfeatures]

        boxplot_data_sharpe = []
        boxplot_data_hjd = []
        positions_sharpe = []
        positions_hjd = []
        labels = []

        base_pos = nfeatures
        for i, kappa in enumerate(kappa_values):
            kappa_data = factor_data[factor_data['kappa'] == kappa]
            if len(kappa_data) > 0:
                boxplot_data_sharpe.append(kappa_data['sharpe'].values)
                boxplot_data_hjd.append(kappa_data['hjd'].values)
                # Stagger positions based on kappa index
                offset = (i - len(kappa_values)/2) * 0.15
                positions_sharpe.append(base_pos + offset)
                positions_hjd.append(base_pos + offset)
                if nfeatures == DKKM_NFEATURES[0]:  # Only create labels once
                    labels.append(f'$\\kappa$={kappa:.4g}')

    # This approach is getting complex. Let's simplify by creating separate plots per metric
    # and using seaborn for better categorical handling

    # Alternative: create one figure per model showing all kappa-factor combinations
    # Let's use a simpler approach: show boxplots for selected kappa values only

    # Select a subset of kappa values to avoid overcrowding
    selected_kappas = kappa_values[:4] if len(kappa_values) > 4 else kappa_values

    boxplot_data_sharpe = []
    boxplot_data_hjd = []
    labels = []

    for nfeatures in DKKM_NFEATURES:
        for kappa in selected_kappas:
            kappa_factor_data = panel_means[(panel_means['kappa'] == kappa) &
                                           (panel_means['num_factors'] == nfeatures)]
            if len(kappa_factor_data) > 0:
                boxplot_data_sharpe.append(kappa_factor_data['sharpe'].values)
                boxplot_data_hjd.append(kappa_factor_data['hjd'].values)
                labels.append(f"$\\kappa$={kappa:.4g}\nn={nfeatures}")

    # Left panel: Sharpe ratios
    bp1 = ax1.boxplot(boxplot_data_sharpe, labels=labels, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.set_title(f'(a) DKKM Sharpe Ratios - {model.upper()}')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    # Right panel: HJ distances
    bp2 = ax2.boxplot(boxplot_data_hjd, labels=labels, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    ax2.set_ylabel('Hansen-Jagannathan Distance', fontsize=12)
    ax2.set_title(f'(b) DKKM HJ Distances - {model.upper()}')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"    Saved to {output_path}")
    plt.close()


def main():
    """Main analysis function."""
    print("="*70)
    print("NOIPCA ANALYSIS")
    print("="*70)
    print()

    print(f"Output directories:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Data directory: {DATA_DIR}")
    print()

    # Discover all panels for each model
    print("Discovering panels...")
    all_panels = {}
    for model in MODELS:
        panels = discover_panels(model)
        all_panels[model] = panels
        print(f"  {model.upper()}: {len(panels)} panel(s) - {panels}")
    print()

    # Load all results
    print("Loading results...")
    fama_all = []
    dkkm_all = []
    ipca_all = []

    for model in MODELS:
        if not all_panels[model]:
            print(f"  {model.upper()}: No panels found, skipping")
            continue

        print(f"  {model.upper()}:")

        # Load Fama
        fama = load_fama_results(model, all_panels[model])
        if len(fama) > 0:
            print(f"    Fama: {len(fama)} observations from {fama['panel'].nunique()} panel(s)")
            fama_all.append(fama)

        # Load DKKM
        dkkm = load_dkkm_results(model, all_panels[model])
        if len(dkkm) > 0:
            print(f"    DKKM: {len(dkkm)} observations from {dkkm['panel'].nunique()} panel(s)")
            dkkm_all.append(dkkm)

        # Load IPCA
        ipca = load_ipca_results(model, all_panels[model])
        if len(ipca) > 0:
            print(f"    IPCA: {len(ipca)} observations from {ipca['panel'].nunique()} panel(s)")
            ipca_all.append(ipca)

    # Combine all models
    fama_all = pd.concat(fama_all, ignore_index=True) if fama_all else pd.DataFrame()
    dkkm_all = pd.concat(dkkm_all, ignore_index=True) if dkkm_all else pd.DataFrame()
    ipca_all = pd.concat(ipca_all, ignore_index=True) if ipca_all else pd.DataFrame()

    print()
    print("Processing data...")

    if len(fama_all) > 0:
        # Rename columns and calculate metrics
        fama_all = fama_all.rename(columns={"mean": "mn"})
        fama_all["sharpe"] = fama_all.mn / fama_all.stdev

        # Filter to alpha=0 for Fama (no shrinkage)
        if 'alpha' in fama_all.columns:
            fama_all = fama_all[fama_all.alpha == 0].copy()

    if len(dkkm_all) > 0:
        # Rename columns and calculate metrics
        dkkm_all = dkkm_all.rename(columns={"alpha": "kappa", "mean": "mn"})
        dkkm_all["sharpe"] = dkkm_all.mn / dkkm_all.stdev

    # Create tables
    print()
    print("Creating tables...")
    if len(fama_all) > 0:
        create_fama_table(fama_all, TABLES_DIR / "fama_comparisons.tex")

    if len(dkkm_all) > 0 and len(fama_all) > 0:
        create_dkkm_table(dkkm_all, fama_all, TABLES_DIR / "dkkm_comparisons.tex")

    # Create figures
    print()
    print("Creating figures...")
    for model in MODELS:
        model_dkkm = dkkm_all[dkkm_all['model_type'] == model] if len(dkkm_all) > 0 else pd.DataFrame()
        model_fama = fama_all[fama_all['model_type'] == model] if len(fama_all) > 0 else pd.DataFrame()

        if len(model_dkkm) > 0 and len(model_fama) > 0:
            ffc = model_fama[model_fama['method'] == 'ff'].copy()
            fmr = model_fama[model_fama['method'] == 'fm'].copy()

            if len(ffc) > 0 and len(fmr) > 0:
                create_dkkm_figure(model_dkkm, ffc, fmr,
                                 FIGURES_DIR / f"dkkm_{model}.pdf", model=model)

    # Create boxplots
    print()
    print("Creating boxplots...")
    if len(fama_all) > 0:
        create_fama_boxplots(fama_all, FIGURES_DIR / "fama_boxplots.pdf")

    for model in MODELS:
        model_dkkm = dkkm_all[dkkm_all['model_type'] == model] if len(dkkm_all) > 0 else pd.DataFrame()
        if len(model_dkkm) > 0:
            create_dkkm_boxplots(model_dkkm, FIGURES_DIR / f"dkkm_boxplots_{model}.pdf", model=model)

    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print(f"Tables: {TABLES_DIR}/")
    print(f"Figures: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
