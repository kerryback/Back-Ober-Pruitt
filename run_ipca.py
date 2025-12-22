"""
IPCA (Instrumented Principal Component Analysis) factor computation.

Estimates latent factors using Stiefel manifold optimization with pymanopt.

Usage:
    python run_ipca.py [panel_id] [K]

Arguments:
    panel_id: Identifier for panel data (e.g., "bgn_0", "kp14_5")
              Reads from {panel_id}_panel.pkl
    K: Number of latent factors (e.g., 1, 2, 3)

Output:
    output/{panel_id}_ipca_{K}.pkl

Examples:
    python run_ipca.py bgn_0 1
    python run_ipca.py kp14_5 3
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
    from config import DATA_DIR
    from utils_factors import ipca_functions as ipca
    from utils_factors import portfolio_stats
    from utils_factors import factor_utils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Back-Ober-Pruitt directory:")
    print("  cd Back-Ober-Pruitt")
    print("  python run_ipca.py")
    sys.exit(1)


def compute_ipca_factors(panel: pd.DataFrame, start: int, end: int, K: int, CONFIG, MODEL, CHARS):
    """
    Compute IPCA (Instrumented Principal Component Analysis) factors.

    Args:
        panel: Panel data
        start: Start month
        end: End month
        K: Number of latent factors
        CONFIG: Model configuration
        MODEL: Model name
        CHARS: List of characteristics

    Returns:
        ipca_weights: (K, N, n_windows) array of factor loadings
        ipca_pi: (K, n_windows) array of portfolio weights
        info_list: List of convergence info dicts
    """
    print(f"\n{'-'*70}")
    print(f"Computing IPCA factors...")
    print(f"  Latent factors: {K}")
    print(f"  Random restarts: {CONFIG['ipca_n_restarts']}")
    print(f"  Max iterations: {CONFIG['ipca_max_iterations']}")
    print(f"  Sign normalize: {CONFIG['ipca_sign_normalize']}")
    print(f"  Warm start: {CONFIG['ipca_warm_start']}")

    t0 = time.time()

    # Get actual number of firms from panel
    N = panel.groupby(level='month').size().max()

    # Compute IPCA using rolling windows
    ipca_weights, ipca_pi, info_list = ipca.fit_ipca_rolling(
        panel=panel,
        K=K,
        N=N,
        start=start,
        end=end,
        chars=CHARS,
        n_restarts=CONFIG['ipca_n_restarts'],
        verbosity=CONFIG['ipca_verbosity']
    )

    elapsed = time.time() - t0
    print(f"[OK] IPCA factors computed in {elapsed:.1f}s at "
          f"{datetime.now().strftime('%I:%M%p')}")

    # Print convergence statistics
    avg_iters = np.mean([info['iterations'] for info in info_list])
    print(f"  Average iterations: {avg_iters:.1f}")
    print(f"  Windows computed: {len(info_list)}")

    return ipca_weights, ipca_pi, info_list


def compute_ipca_factor_returns(panel: pd.DataFrame, ipca_weights: np.ndarray,
                                 start: int, end: int, K: int):
    """
    Compute IPCA factor returns from loadings.

    Args:
        panel: Panel data
        ipca_weights: (K, N, n_windows) factor loadings
        start: Start month
        end: End month
        K: Number of factors

    Returns:
        DataFrame of factor returns (months x K factors)
    """
    print(f"\n{'-'*70}")
    print("Computing IPCA factor returns...")

    n_windows = ipca_weights.shape[2]
    factor_returns = []

    for i, t in enumerate(range(start + 360, end + 1)):
        if i >= n_windows:
            break

        data = panel.loc[t]
        firms = data.index.to_numpy()
        r_t = data['xret'].to_numpy()

        # Get loadings for this month
        loadings_t = ipca_weights[:, firms, i]  # (K, N_t)

        # Factor return = loading' @ returns
        f_t = loadings_t @ r_t  # (K,)

        factor_returns.append(f_t)

    # Create DataFrame
    ipca_factors = pd.DataFrame(
        factor_returns,
        index=range(start + 360, start + 360 + len(factor_returns)),
        columns=[f'ipca_{k+1}' for k in range(K)]
    )
    ipca_factors.index.name = 'month'

    print(f"[OK] Factor returns: {ipca_factors.shape}")
    return ipca_factors


def main():
    """Main execution function."""
    start_time = time.time()

    # Parse command-line arguments
    panel_id, model_name, parsed_args = factor_utils.parse_panel_arguments(
        script_name='run_ipca',
        additional_args={'K': 'int'}
    )

    # Load model configuration
    CONFIG = factor_utils.load_model_config(model_name)

    # Get K from command line (required)
    if 'K' not in parsed_args:
        print("ERROR: K argument is required")
        print("\nUsage: python run_ipca.py [panel_id] [K]")
        print("  Example: python run_ipca.py bgn_0 1")
        sys.exit(1)

    K = parsed_args['K']

    # Validate K
    L = len(CONFIG['chars']) + 1  # +1 for intercept
    if K > L:
        print(f"ERROR: K={K} cannot exceed L={L} (number of characteristics + 1)")
        print(f"  Model {model_name} has {len(CONFIG['chars'])} characteristics")
        print(f"  Maximum K = {L}")
        sys.exit(1)

    # Model selection
    MODEL = CONFIG['model']
    CHARS = CONFIG['chars']

    # Load panel data
    panel, arrays_data = factor_utils.load_panel_data(panel_id, model_name)

    # Prepare panel
    panel, start, end = factor_utils.prepare_panel(panel, CHARS)

    # Extract actual N and T from loaded data
    CONFIG['T'] = end - start + 1  # Actual number of periods
    CONFIG['N'] = panel.groupby(level='month').size().max()  # Max firms per month

    # Print header
    factor_utils.print_script_header(
        title="IPCA (INSTRUMENTED PRINCIPAL COMPONENT ANALYSIS)",
        model=MODEL,
        panel_id=panel_id,
        config=CONFIG,
        additional_info={'K (Latent Factors)': K}
    )

    # Check pymanopt availability
    if not ipca.PYMANOPT_AVAILABLE:
        print("\nERROR: pymanopt is required for IPCA but not installed")
        print("Install with: pip install pymanopt>=2.2.0")
        sys.exit(1)

    # Compute IPCA factors
    ipca_weights, ipca_pi, info_list = compute_ipca_factors(
        panel, start, end, K, CONFIG, MODEL, CHARS
    )

    # Compute factor returns
    ipca_factors = compute_ipca_factor_returns(
        panel, ipca_weights, start, end, K
    )

    # Compute portfolio statistics
    print(f"\n{'-'*70}")
    print("Computing portfolio statistics...")
    print(f"  Alpha grid: {CONFIG['ipca_alpha_lst']}")
    print(f"  Include market: {CONFIG['ipca_include_mkt']}")

    t0 = time.time()

    ipca_stats = None
    if ipca_factors is not None:
        # Get market returns if needed
        mkt_returns = None
        if CONFIG['ipca_include_mkt'] and 'arrays' in arrays_data:
            # Extract market returns from arrays (if available)
            arrays = arrays_data['arrays']
            if 'mkt_rf' in arrays:
                mkt_returns = pd.DataFrame(
                    {'mkt_rf': arrays['mkt_rf']},
                    index=range(start, end + 1)
                )
                mkt_returns.index.name = 'month'

        ipca_stats = portfolio_stats.compute_ipca_portfolio_stats(
            ipca_factors,
            panel,
            panel_id=panel_id,
            model=model_name,
            K=K,
            chars=CHARS,
            start_month=start,
            end_month=end,
            alpha_lst=CONFIG['ipca_alpha_lst'],
            include_mkt=CONFIG['ipca_include_mkt'],
            mkt_returns=mkt_returns,
            burnin=CONFIG['burnin']
        )
        print(f"  [OK] IPCA stats: {len(ipca_stats)} observations")

    elapsed = time.time() - t0
    print(f"[OK] Portfolio statistics computed in {elapsed:.1f}s")

    # Summary
    print(f"\nFactor Returns Summary:")
    print(f"  IPCA factors: {ipca_factors.shape}")
    print(f"  Latent factors (K): {K}")
    print(f"  Windows: {len(info_list)}")

    # Create results dictionary
    results = {
        'ipca_factors': ipca_factors,
        'ipca_weights': ipca_weights,
        'ipca_pi': ipca_pi,
        'ipca_stats': ipca_stats,
        'info_list': info_list,
        'K': K,
        'panel_id': panel_id,
        'model': MODEL,
        'chars': CHARS,
        'start': start,
        'end': end,
    }

    # Save results
    output_file = os.path.join(DATA_DIR, f"{panel_id}_ipca_{K}.pkl")
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
            "ipca = results['ipca_factors']",
        ]
    )


if __name__ == "__main__":
    main()
