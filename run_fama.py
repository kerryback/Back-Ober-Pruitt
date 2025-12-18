"""
Fama-French and Fama-MacBeth factor computation.

Computes only:
1. Fama-French factors (characteristic-sorted portfolios)
2. Fama-MacBeth factors (two-stage cross-sectional regression)

Usage:
    python run_fama.py [panel_id]

Arguments:
    panel_id: Identifier for panel data (e.g., "bgn_0", "kp14_5")
              Reads from {panel_id}_arrays.pkl
              Output: output/{panel_id}_fama.pkl

Examples:
    python run_fama.py bgn_0
    python run_fama.py kp14_5
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
import time

# Import refactored modules
try:
    from config import LOADING_KEYS, FACTOR_KEYS, DATA_DIR
    from utils_factors import fama_functions as fama
    from utils_factors import portfolio_stats
    from utils_factors import factor_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Back-Ober-Pruitt directory:")
    print("  cd Back-Ober-Pruitt")
    print("  python run_fama.py")
    sys.exit(1)


def compute_fama_factors(panel: pd.DataFrame, start: int, end: int, CONFIG, CHARS):
    """Compute Fama-French and Fama-MacBeth factors."""
    print(f"\n{'-'*70}")
    print("Computing Fama-French and Fama-MacBeth factors...")

    t0 = time.time()

    ff_rets = fama.factors(fama.fama_french, panel,
                           n_jobs=CONFIG['n_jobs'], start=start, end=end, chars=CHARS)
    fm_rets = fama.factors(fama.fama_macbeth, panel,
                           n_jobs=CONFIG['n_jobs'], start=start, end=end, chars=CHARS,
                           stdz_fm=CONFIG['stdz_fm'])

    elapsed = time.time() - t0
    print(f"[OK] Fama factors computed in {elapsed:.1f}s at "
          f"{datetime.now().strftime('%I:%M%p')}")

    return ff_rets, fm_rets


def main():
    """Main execution function."""
    start_time = time.time()

    # Parse command-line arguments
    panel_id, model_name, _ = factor_utils.parse_panel_arguments(script_name='run_fama')

    # Load model configuration
    CONFIG = factor_utils.load_model_config(model_name)

    # Model selection
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']
    NAMES = CONFIG['factor_names']

    # Load panel data
    panel, arrays_data = factor_utils.load_panel_data(panel_id, model_name)

    # Prepare panel
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    # Extract actual N and T from loaded data
    CONFIG['T'] = end - start + 1  # Actual number of periods
    CONFIG['N'] = panel.groupby(level='month').size().max()  # Max firms per month

    # Print header
    factor_utils.print_script_header(
        title="FAMA-FRENCH & FAMA-MACBETH FACTORS",
        model=MODEL,
        panel_id=panel_id,
        config=CONFIG,
        additional_info={'Methods': 'Fama-French, Fama-MacBeth'}
    )

    # Compute Fama factors
    ff_rets, fm_rets = compute_fama_factors(panel, start, end, CONFIG, CHARS)

    # Extract model factors from arrays data
    print(f"\n{'-'*70}")
    print("Extracting model factors from arrays data...")

    model_premia = {}
    if 'arrays' in arrays_data:
        arrays = arrays_data['arrays']

        # Extract Taylor and Projection factor returns
        if 'taylor' in FACTOR_KEYS.get(MODEL, []) or 'f_1_' in arrays:
            taylor_keys = [k for k in arrays.keys() if k.startswith('f_1_')]
            if taylor_keys:
                taylor_data = {k.replace('f_1_', ''): arrays[k] for k in taylor_keys}
                model_premia['taylor'] = pd.DataFrame(taylor_data, index=range(start, end + 1))
                model_premia['taylor'].index.name = 'month'
                print(f"  [OK] Taylor factors: {model_premia['taylor'].shape}")

        if 'proj' in FACTOR_KEYS.get(MODEL, []) or 'f_2_' in arrays:
            proj_keys = [k for k in arrays.keys() if k.startswith('f_2_')]
            if proj_keys:
                proj_data = {k.replace('f_2_', ''): arrays[k] for k in proj_keys}
                model_premia['proj'] = pd.DataFrame(proj_data, index=range(start, end + 1))
                model_premia['proj'].index.name = 'month'
                print(f"  [OK] Projection factors: {model_premia['proj'].shape}")

    # Compute portfolio statistics
    print(f"\n{'-'*70}")
    print("Computing portfolio statistics...")
    print(f"  Alpha grids: Fama={CONFIG['alpha_lst_fama']}")

    t0 = time.time()

    # Model factors statistics (if available)
    model_stats = None
    if model_premia:
        model_stats = portfolio_stats.compute_model_portfolio_stats(
            model_premia, panel, start, end
        )
        print(f"  [OK] Model stats: {len(model_stats)} observations")

    # Fama factors statistics
    fama_stats = portfolio_stats.compute_fama_portfolio_stats(
        ff_rets, fm_rets, panel,
        panel_id=panel_id,
        model=model_name,
        chars=CHARS,
        start_month=start,
        end_month=end,
        alpha_lst=CONFIG['alpha_lst_fama'],
        burnin=CONFIG['burnin']
    )
    print(f"  [OK] Fama stats: {len(fama_stats)} observations")

    elapsed = time.time() - t0
    print(f"[OK] Portfolio statistics computed in {elapsed:.1f}s")

    # Summary
    print(f"\nFactor Returns Summary:")
    print(f"  FF returns shape: {ff_rets.shape}")
    print(f"  FM returns shape: {fm_rets.shape}")
    if model_premia:
        for method, rets in model_premia.items():
            print(f"  Model {method} returns shape: {rets.shape}")

    # Create results dictionary
    results = {
        'ff_returns': ff_rets,
        'fm_returns': fm_rets,
        'fama_stats': fama_stats,
        'model_premia_taylor': model_premia.get('taylor'),
        'model_premia_proj': model_premia.get('proj'),
        'model_stats': model_stats,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'start': start,
        'end': end,
    }

    # Save results
    output_file = os.path.join(DATA_DIR, f"{panel_id}_fama.pkl")
    factor_utils.save_factor_results(results, output_file, verbose=True)

    # Print runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    # Print footer with usage examples
    factor_utils.print_script_footer(
        panel_id=panel_id,
        usage_examples=[
            "import pickle",
            f"with open('{output_file}', 'rb') as f:",
            "    results = pickle.load(f)",
            "ff = results['ff_returns']",
            "fm = results['fm_returns']",
        ]
    )


if __name__ == "__main__":
    main()
