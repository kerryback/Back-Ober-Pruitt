"""
Analysis script to produce LaTeX tables from Back-Ober-Pruitt results.

This script automatically discovers and analyzes all panels for each model (BGN, KP14, GS21),
following the recipe from analysis.ipynb:
- Compute sharpe = mean / stdev for each month
- Compute hjd_realized = (mean - xret)^2 for each month
- Aggregate by panel, then across panels

Creates 15 LaTeX tables (5 for each model):
1. Fama table (FFC/FMR sharpe and hjd)
2-3. DKKM sharpe and hjd tables
4-5. IPCA sharpe and hjd tables

Usage:
    python analysis.py

Outputs:
    - LaTeX tables saved to tables/
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR

# Output directory
SCRIPT_DIR = Path(__file__).parent
TABLES_DIR = SCRIPT_DIR / "tables"

# Create output directory if it doesn't exist
TABLES_DIR.mkdir(exist_ok=True)

# Configuration
MODELS = ['bgn', 'kp14', 'gs21']


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


def load_and_process_fama(model: str, panel_indices: List[int]) -> pd.DataFrame:
    """
    Load Fama results and compute sharpe/hjd following the notebook recipe.

    For each month:
    - sharpe = mean / stdev
    - hjd_realized = (mean - xret)^2

    Returns DataFrame with panel-level aggregates.
    """
    all_data = []

    for panel_idx in panel_indices:
        panel_id = f"{model}_{panel_idx}"
        fama_file = os.path.join(DATA_DIR, f"{panel_id}_fama.pkl")

        if not os.path.exists(fama_file):
            continue

        with open(fama_file, 'rb') as f:
            fama_data = pickle.load(f)

        fama_stats = fama_data.get('fama_stats')
        if fama_stats is None or len(fama_stats) == 0:
            continue

        # Compute sharpe and hjd_realized for each month
        fama_stats = fama_stats.copy()
        fama_stats['sharpe'] = fama_stats['mean'] / fama_stats['stdev']
        fama_stats['hjd_realized'] = (fama_stats['mean'] - fama_stats['xret'])**2

        # Group by (panel, method, alpha) and compute panel-level statistics
        panel_stats = fama_stats.groupby(['method', 'alpha']).agg({
            'sharpe': 'mean',  # Mean sharpe across months
            'hjd_realized': lambda x: np.sqrt(x.mean())  # sqrt(mean(hjd_realized))
        }).reset_index()

        panel_stats['panel'] = panel_idx
        panel_stats.rename(columns={'hjd_realized': 'hjd'}, inplace=True)
        all_data.append(panel_stats)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def load_and_process_dkkm(model: str, panel_indices: List[int]) -> pd.DataFrame:
    """
    Load DKKM results and compute sharpe/hjd following the notebook recipe.

    Returns DataFrame with panel-level aggregates.
    """
    all_data = []

    for panel_idx in panel_indices:
        panel_id = f"{model}_{panel_idx}"

        # Get all DKKM files for this panel
        pattern = os.path.join(DATA_DIR, f"{panel_id}_dkkm_*.pkl")
        for filepath in glob.glob(pattern):
            # Extract nfeatures from filename
            filename = Path(filepath).stem
            parts = filename.split('_')
            if len(parts) < 4:
                continue

            try:
                nfeatures = int(parts[3])
            except ValueError:
                continue

            with open(filepath, 'rb') as f:
                dkkm_data = pickle.load(f)

            dkkm_stats = dkkm_data.get('dkkm_stats')
            if dkkm_stats is None or len(dkkm_stats) == 0:
                continue

            # Compute sharpe and hjd_realized for each month
            dkkm_stats = dkkm_stats.copy()
            dkkm_stats['sharpe'] = dkkm_stats['mean'] / dkkm_stats['stdev']
            dkkm_stats['hjd_realized'] = (dkkm_stats['mean'] - dkkm_stats['xret'])**2

            # Group by (panel, alpha, nfeatures) and compute panel-level statistics
            panel_stats = dkkm_stats.groupby(['alpha']).agg({
                'sharpe': 'mean',
                'hjd_realized': lambda x: np.sqrt(x.mean())
            }).reset_index()

            panel_stats['panel'] = panel_idx
            panel_stats['num_factors'] = nfeatures
            panel_stats.rename(columns={'hjd_realized': 'hjd'}, inplace=True)
            all_data.append(panel_stats)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def load_and_process_ipca(model: str, panel_indices: List[int]) -> pd.DataFrame:
    """
    Load IPCA results and compute sharpe/hjd following the notebook recipe.

    Returns DataFrame with panel-level aggregates.
    """
    all_data = []

    for panel_idx in panel_indices:
        panel_id = f"{model}_{panel_idx}"

        # Get all IPCA files for this panel
        pattern = os.path.join(DATA_DIR, f"{panel_id}_ipca_*.pkl")
        for filepath in glob.glob(pattern):
            # Extract K from filename
            filename = Path(filepath).stem
            parts = filename.split('_')
            if len(parts) < 4:
                continue

            try:
                K = int(parts[3])
            except ValueError:
                continue

            with open(filepath, 'rb') as f:
                ipca_data = pickle.load(f)

            ipca_stats = ipca_data.get('ipca_stats')
            if ipca_stats is None or len(ipca_stats) == 0:
                continue

            # Compute sharpe and hjd_realized for each month
            ipca_stats = ipca_stats.copy()
            ipca_stats['sharpe'] = ipca_stats['mean'] / ipca_stats['stdev']
            ipca_stats['hjd_realized'] = (ipca_stats['mean'] - ipca_stats['xret'])**2

            # Group by (panel, alpha, K) and compute panel-level statistics
            panel_stats = ipca_stats.groupby(['alpha']).agg({
                'sharpe': 'mean',
                'hjd_realized': lambda x: np.sqrt(x.mean())
            }).reset_index()

            panel_stats['panel'] = panel_idx
            panel_stats['num_factors'] = K
            panel_stats.rename(columns={'hjd_realized': 'hjd'}, inplace=True)
            all_data.append(panel_stats)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def create_fama_table(fama_df: pd.DataFrame, model: str, output_path: str):
    """
    Create Fama table for a single model.

    Rows: alpha values
    Columns: (FFC, sharpe), (FMR, sharpe), (FFC, hjd), (FMR, hjd)
    Values: Mean across panels
    """
    if len(fama_df) == 0:
        print(f"  WARNING: No Fama data for {model}")
        return

    # Get unique alpha values
    alphas = sorted(fama_df['alpha'].unique())

    # Create table data
    table_data = []
    for alpha in alphas:
        alpha_data = fama_df[fama_df['alpha'] == alpha]

        row = {'alpha': alpha}

        # FFC sharpe
        ffc_data = alpha_data[alpha_data['method'] == 'ff']
        row['FFC_sharpe'] = ffc_data['sharpe'].mean() if len(ffc_data) > 0 else np.nan

        # FMR sharpe
        fmr_data = alpha_data[alpha_data['method'] == 'fm']
        row['FMR_sharpe'] = fmr_data['sharpe'].mean() if len(fmr_data) > 0 else np.nan

        # FFC hjd
        row['FFC_hjd'] = ffc_data['hjd'].mean() if len(ffc_data) > 0 else np.nan

        # FMR hjd
        row['FMR_hjd'] = fmr_data['hjd'].mean() if len(fmr_data) > 0 else np.nan

        table_data.append(row)

    df = pd.DataFrame(table_data)
    df = df.set_index('alpha')

    # Create LaTeX table
    latex = df.to_latex(float_format="%.4f", na_rep="--",
                        column_format='r' + 'r'*len(df.columns),
                        escape=False)

    # Add caption and label
    caption = f"""\\caption{{\\textbf{{Fama-French Performance - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show mean Sharpe ratio and
    Hansen-Jagannathan distance across panels for FFC and FMR methods.}}
\\label{{tab:fama_{model}}}"""

    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"  Saved {output_path}")


def create_dkkm_tables(dkkm_df: pd.DataFrame, model: str, sharpe_path: str, hjd_path: str):
    """
    Create DKKM sharpe and hjd tables for a single model.

    Rows: alpha values
    Columns: num_factors
    Values: Mean across panels
    """
    if len(dkkm_df) == 0:
        print(f"  WARNING: No DKKM data for {model}")
        return

    # Get unique alpha and num_factors values
    alphas = sorted(dkkm_df['alpha'].unique())
    num_factors_vals = sorted(dkkm_df['num_factors'].unique())

    # Create sharpe table
    sharpe_table = []
    for alpha in alphas:
        row = {'alpha': alpha}
        alpha_data = dkkm_df[dkkm_df['alpha'] == alpha]
        for nf in num_factors_vals:
            nf_data = alpha_data[alpha_data['num_factors'] == nf]
            row[nf] = nf_data['sharpe'].mean() if len(nf_data) > 0 else np.nan
        sharpe_table.append(row)

    sharpe_df = pd.DataFrame(sharpe_table).set_index('alpha')

    # Create hjd table
    hjd_table = []
    for alpha in alphas:
        row = {'alpha': alpha}
        alpha_data = dkkm_df[dkkm_df['alpha'] == alpha]
        for nf in num_factors_vals:
            nf_data = alpha_data[alpha_data['num_factors'] == nf]
            row[nf] = nf_data['hjd'].mean() if len(nf_data) > 0 else np.nan
        hjd_table.append(row)

    hjd_df = pd.DataFrame(hjd_table).set_index('alpha')

    # Save sharpe table
    latex = sharpe_df.to_latex(float_format="%.4f", na_rep="--",
                                column_format='r' + 'r'*len(sharpe_df.columns),
                                escape=False)
    caption = f"""\\caption{{\\textbf{{DKKM Sharpe Ratios - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show number of DKKM factors.
    Values are mean Sharpe ratios across panels.}}
\\label{{tab:dkkm_sharpe_{model}}}"""
    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(sharpe_path, 'w') as f:
        f.write(latex)
    print(f"  Saved {sharpe_path}")

    # Save hjd table
    latex = hjd_df.to_latex(float_format="%.4f", na_rep="--",
                             column_format='r' + 'r'*len(hjd_df.columns),
                             escape=False)
    caption = f"""\\caption{{\\textbf{{DKKM Hansen-Jagannathan Distances - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show number of DKKM factors.
    Values are mean HJ distances across panels.}}
\\label{{tab:dkkm_hjd_{model}}}"""
    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(hjd_path, 'w') as f:
        f.write(latex)
    print(f"  Saved {hjd_path}")


def create_ipca_tables(ipca_df: pd.DataFrame, model: str, sharpe_path: str, hjd_path: str):
    """
    Create IPCA sharpe and hjd tables for a single model.

    Rows: alpha values
    Columns: num_factors (K values)
    Values: Mean across panels
    """
    if len(ipca_df) == 0:
        print(f"  WARNING: No IPCA data for {model}")
        return

    # Get unique alpha and num_factors values
    alphas = sorted(ipca_df['alpha'].unique())
    num_factors_vals = sorted(ipca_df['num_factors'].unique())

    # Create sharpe table
    sharpe_table = []
    for alpha in alphas:
        row = {'alpha': alpha}
        alpha_data = ipca_df[ipca_df['alpha'] == alpha]
        for K in num_factors_vals:
            K_data = alpha_data[alpha_data['num_factors'] == K]
            row[K] = K_data['sharpe'].mean() if len(K_data) > 0 else np.nan
        sharpe_table.append(row)

    sharpe_df = pd.DataFrame(sharpe_table).set_index('alpha')

    # Create hjd table
    hjd_table = []
    for alpha in alphas:
        row = {'alpha': alpha}
        alpha_data = ipca_df[ipca_df['alpha'] == alpha]
        for K in num_factors_vals:
            K_data = alpha_data[alpha_data['num_factors'] == K]
            row[K] = K_data['hjd'].mean() if len(K_data) > 0 else np.nan
        hjd_table.append(row)

    hjd_df = pd.DataFrame(hjd_table).set_index('alpha')

    # Save sharpe table
    latex = sharpe_df.to_latex(float_format="%.4f", na_rep="--",
                                column_format='r' + 'r'*len(sharpe_df.columns),
                                escape=False)
    caption = f"""\\caption{{\\textbf{{IPCA Sharpe Ratios - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show number of IPCA factors (K).
    Values are mean Sharpe ratios across panels.}}
\\label{{tab:ipca_sharpe_{model}}}"""
    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(sharpe_path, 'w') as f:
        f.write(latex)
    print(f"  Saved {sharpe_path}")

    # Save hjd table
    latex = hjd_df.to_latex(float_format="%.4f", na_rep="--",
                             column_format='r' + 'r'*len(hjd_df.columns),
                             escape=False)
    caption = f"""\\caption{{\\textbf{{IPCA Hansen-Jagannathan Distances - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show number of IPCA factors (K).
    Values are mean HJ distances across panels.}}
\\label{{tab:ipca_hjd_{model}}}"""
    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(hjd_path, 'w') as f:
        f.write(latex)
    print(f"  Saved {hjd_path}")


def main():
    """Main analysis function."""
    print("="*70)
    print("ANALYSIS - Creating LaTeX Tables")
    print("="*70)
    print()

    for model in MODELS:
        print(f"Processing {model.upper()}...")

        # Discover panels
        panel_indices = discover_panels(model)
        if not panel_indices:
            print(f"  WARNING: No panels found for {model}")
            print()
            continue

        print(f"  Found {len(panel_indices)} panels: {panel_indices}")

        # Load and process data
        print("  Loading Fama data...")
        fama_df = load_and_process_fama(model, panel_indices)

        print("  Loading DKKM data...")
        dkkm_df = load_and_process_dkkm(model, panel_indices)

        print("  Loading IPCA data...")
        ipca_df = load_and_process_ipca(model, panel_indices)

        # Create tables
        print("  Creating tables...")

        # Table 1: Fama
        fama_path = TABLES_DIR / f"fama_{model}.tex"
        create_fama_table(fama_df, model, str(fama_path))

        # Tables 2-3: DKKM
        dkkm_sharpe_path = TABLES_DIR / f"dkkm_sharpe_{model}.tex"
        dkkm_hjd_path = TABLES_DIR / f"dkkm_hjd_{model}.tex"
        create_dkkm_tables(dkkm_df, model, str(dkkm_sharpe_path), str(dkkm_hjd_path))

        # Tables 4-5: IPCA
        ipca_sharpe_path = TABLES_DIR / f"ipca_sharpe_{model}.tex"
        ipca_hjd_path = TABLES_DIR / f"ipca_hjd_{model}.tex"
        create_ipca_tables(ipca_df, model, str(ipca_sharpe_path), str(ipca_hjd_path))

        print()

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print(f"Tables: {TABLES_DIR}/")


if __name__ == "__main__":
    main()
