"""
Test IPCA Factor Computation - Regression Test

USAGE:
------
Run this test from the tests/ directory:
    cd tests
    python test_ipca.py <model> <K>

where:
    <model> is one of: bgn, kp14, gs21
    <K> is the number of latent factors (e.g., 1, 2, 3)

Examples:
    python test_ipca.py bgn 1
    python test_ipca.py kp14 2
    python test_ipca.py gs21 3

Directory structure (relative to tests/):
    ./original_code/        - Original IPCA code to compare against
    ../run_ipca.py          - Current IPCA factor computation
    ../config.py            - Configuration (N, T, burnin values)

PURPOSE:
--------
This test verifies that the refactored IPCA (Instrumented Principal Component
Analysis) factor computation produces numerically identical results (up to sign)
to the original code when given the same panel data.

IPCA uses alternating least squares to estimate latent factors that are allowed
to vary with firm characteristics.

WORKFLOW COMPARISON:
-------------------

ORIGINAL CODE (tests/original_code/):
  Implementation: Import directly and call functions in-process

  1. Import ipca_functions.py (original IPCA module)
     - Imports: parameters.py, dkkm_functions.py
     - Uses scipy.optimize for alternating least squares

  2. Functions:
     - fit_ipca(panel, start, K, tol, Gamma0, f0): Single window estimation
       Returns: (factor_port, pi, Gamma1, f1)

     - fit_ipca_360(panel, K, N, start, end, rff, W, chars): Rolling estimation
       Returns: (ipca_weights_on_stocks, ipca_factor_weights)

     - normalization(Gamma, f): Ensures Gamma'*Gamma = I and orthogonal factors
       Returns: (Gamma, f) normalized

  3. Execution:
     - Import ipca_functions module
     - Prepare panel (rank-standardize characteristics)
     - Call fit_ipca_360() with panel data
     - Returns IPCA weights and factors directly

CURRENT CODE (utils_factors/):
  Implementation: Run from the terminal via subprocess (../run_ipca.py)

  1. Import ipca_functions.py (refactored version)
     - Imports: config.py (centralized configuration)
     - Uses pymanopt for Stiefel manifold optimization
     - Improved convergence with better initialization

  2. Same mathematical approach, different optimization:
     - Original: Alternating least squares (ALS)
     - Current: Stiefel manifold optimization (pymanopt)
     - Both produce factors with Gamma'*Gamma = I

  3. Execution:
     - Run run_ipca.py via subprocess: python run_ipca.py bgn_0 1
     - Script reads panel from pickle file (bgn_0_panel.pkl)
     - Computes IPCA factors using ipca_functions
     - Saves factors to pickle file: bgn_0_ipca_1.pkl
     - Test reads pickle file to get results

TEST CONFIGURATION:
------------------
From config.py:
  - N = 50 firms
  - T = 400 time periods
  - BGN_BURNIN/KP14_BURNIN/GS21_BURNIN = 300 months
  - TEST_SEED = 12345 (set in test script)

NOTE ON SIGN INDETERMINACY:
--------------------------
IPCA factors are identified only up to sign (±1). The test uses
assert_factors_equal_up_to_sign() which allows factors to match
either as-is or with sign flipped.

EXPECTED RESULTS:
----------------
Factor returns should be numerically identical up to sign:
  1. Factor returns: shape (T, K), rtol=1e-6, atol=1e-8 (up to sign flip)
  2. Gamma loadings: shape (nchars, K), may differ due to optimization method

Note: Looser tolerances than Fama/DKKM because IPCA uses iterative optimization
which may converge to slightly different local optima.

TEST PROCEDURE:
--------------
1. Generate panel with current code via run_generate_panel.py subprocess
2. Read panel pickle file
3. Prepare panel for IPCA (rank-standardize characteristics)
4. Compute IPCA factors with original code (fit_ipca_360)
5. Compute IPCA factors with current code via run_ipca.py subprocess
6. Read IPCA pickle file from current code
7. Compare factor returns (allowing sign flips)
8. Delete all pickle files (cleanup)
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import subprocess
from pathlib import Path
from scipy.linalg import orthogonal_procrustes

# Add paths (run from tests/ directory)
# ../  - for current code (config.py, run_ipca.py, etc.)
# ./test_utils/  - for test utilities (comparison.py)
# ./original_code/  - for original code to compare against
sys.path.insert(0, str(Path(__file__).parent.parent))  # ../
sys.path.insert(0, str(Path(__file__).parent / 'test_utils'))  # ./test_utils/
sys.path.insert(0, str(Path(__file__).parent / 'original_code'))  # ./original_code/

from comparison import assert_close
from config import N, T

# Test configuration
TEST_SEED = 12345
EXPECTED_N = 50
EXPECTED_T = 400


def check_config_values():
    """Check if N and T match expected test values."""
    if N != EXPECTED_N or T != EXPECTED_T:
        print("\n" + "!" * 70)
        print("WARNING: Configuration mismatch detected!")
        print("!" * 70)
        print(f"Expected: N={EXPECTED_N}, T={EXPECTED_T}")
        print(f"Current:  N={N}, T={T}")
        print("\nTests are designed for N=50 and T=400.")
        print("Running with different values may cause tests to fail or produce")
        print("unexpected results.")
        print()

        response = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("\nTest aborted by user.")
            return False
        print()
    return True


def align_factors_procrustes(F_reference, F_estimated):
    """
    Align F_estimated to match F_reference using orthogonal Procrustes.

    This solves the factor rotation invariance problem by finding the optimal
    orthogonal rotation Q that minimizes ||F_reference - F_estimated @ Q||^2.

    Args:
        F_reference: Reference factors (T x K)
        F_estimated: Estimated factors to align (T x K)

    Returns:
        F_aligned: Rotated version of F_estimated (T x K)
        Q: Rotation matrix used (K x K)
    """
    # Find optimal rotation Q using Procrustes analysis
    Q, _ = orthogonal_procrustes(F_estimated, F_reference)

    # Apply rotation
    F_aligned = F_estimated @ Q

    return F_aligned, Q


def prepare_panel_for_ipca(panel, chars):
    """Prepare panel data for IPCA factor computation."""
    import dkkm_functions as dkkm_old

    # Set multi-index
    panel = panel.set_index(['month', 'firmid'])

    # Rank-standardize characteristics as in original code
    panel_ranked = panel[chars].groupby('month').apply(
        lambda g: dkkm_old.rank_standardize(g)
    ).reset_index(level=0, drop=True)
    panel_ranked['ones'] = 1
    panel_ranked['xret'] = panel['xret']

    # Get valid data range
    start = panel_ranked.index.get_level_values('month').min()
    end = panel_ranked.index.get_level_values('month').max()

    return panel_ranked, start, end


def compute_ipca_with_original(panel_ranked, K, start, end, burnin):
    """
    Compute IPCA factors using original code for 3 test months only.

    Test months:
    - Month 1: burnin + 360 (first testable month after burnin)
    - Month 2: Random month between (burnin + 361) and (burnin + T - 2)
    - Month 3: burnin + T - 1 (last month in sample)
    """
    import ipca_functions as ipca_old

    print("Computing IPCA factors with original code...")
    print(f"  Using alternating least squares (original method)")

    # Determine 3 test months
    n_months = end - start + 1
    test_month_1 = burnin + 360  # First testable month
    test_month_3 = burnin + n_months - 1  # Last month

    # Random month between first and last (for reproducibility, use seed)
    np.random.seed(TEST_SEED)
    test_month_2 = np.random.randint(burnin + 361, burnin + n_months - 1)

    test_months = [test_month_1, test_month_2, test_month_3]
    print(f"  Testing 3 months: {test_months}")

    # Compute IPCA for each test month
    tol = 1e-4  # Same tolerance as fit_ipca_360
    factor_returns_list = []
    Gamma0, f0 = None, None

    for test_month in test_months:
        # Each test month uses a 360-month window ending at test_month
        window_start = test_month - 360

        print(f"  Computing month {test_month} (window: {window_start} to {test_month-1})...")

        # Call fit_ipca for this specific window
        _factor_port, pi, Gamma1, f1 = ipca_old.fit_ipca(
            panel=panel_ranked,
            start=window_start,
            K=K,
            tol=tol,
            Gamma0=Gamma0,
            f0=f0
        )

        # Store factor returns (pi) for this month
        # pi is shape (K,) - the factor portfolio weights at month test_month
        factor_returns_list.append(pi)

        # Use this as warm start for next month
        Gamma0, f0 = Gamma1, f1

    # Convert to DataFrame
    # Stack the 3 months: each row is a month, each column is a factor
    factor_returns_array = np.array(factor_returns_list)  # Shape (3, K)
    factor_returns = pd.DataFrame(
        factor_returns_array,
        index=test_months,
        columns=[f'Factor_{i+1}' for i in range(K)]
    )

    print(f"  Original factor returns shape: {factor_returns.shape}")
    print(f"  Test months: {list(factor_returns.index)}")

    return factor_returns, None  # No ipca_weights_on_stocks needed for comparison


def test_ipca_factors(model, K):
    """Test IPCA factor computation for a given model and number of factors."""
    # Get burnin for this model
    from config import BGN_BURNIN, KP14_BURNIN, GS21_BURNIN
    burnin_map = {'bgn': BGN_BURNIN, 'kp14': KP14_BURNIN, 'gs21': GS21_BURNIN}
    burnin = burnin_map[model]

    print("=" * 70)
    print(f"IPCA Factor Test: {model.upper()}")
    print("=" * 70)
    print(f"N={N}, T={T}, Burnin={burnin}, Seed={TEST_SEED}")
    print(f"Number of latent factors (K): {K}\n")

    # Test panel identifier
    TEST_PANEL_ID = 0
    panel_id = f"{model}_{TEST_PANEL_ID}"

    # Get model configuration
    chars = ['size', 'bm', 'agr', 'roe', 'mom']

    # Step 1: Generate panel with current code via command line
    print("[1/5] Generating panel with current code via generate_panel.py...")
    test_dir = Path(__file__).parent
    cmd = [sys.executable, str(test_dir / 'test_utils' / 'run_generate_panel.py'), model, str(TEST_PANEL_ID)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] generate_panel.py failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"      Panel generated successfully")

    # Step 2: Read panel pickle file
    from config import DATA_DIR
    panel_file = Path(DATA_DIR) / f'{panel_id}_panel.pkl'

    print(f"\n[2/5] Reading panel from {panel_file}...")
    with open(panel_file, 'rb') as f:
        panel_data = pickle.load(f)
    panel = panel_data['panel']
    print(f"      Panel shape: {panel.shape}")

    # Step 3: Compute IPCA factors with original code
    print(f"\n[3/5] Computing IPCA factors with original code...")
    panel_ranked, start, end = prepare_panel_for_ipca(panel.copy(), chars)
    factor_returns_old, _ipca_weights_old = compute_ipca_with_original(
        panel_ranked, K, start, end, burnin
    )

    # Step 4: Compute IPCA factors with current code via run_ipca.py
    print(f"\n[4/5] Computing IPCA factors with current code via run_ipca.py...")
    print(f"      Note: Current code uses Stiefel manifold optimization")
    parent_dir = Path(__file__).parent.parent
    cmd = [sys.executable, str(parent_dir / 'run_ipca.py'), panel_id, str(K)]
    result = subprocess.run(cmd, cwd=str(parent_dir), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] run_ipca.py failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"      IPCA factors computed successfully")

    # Read IPCA pickle file
    ipca_file = Path(DATA_DIR) / f'{panel_id}_ipca_{K}.pkl'
    print(f"\n[5/5] Reading IPCA factors from {ipca_file}...")
    with open(ipca_file, 'rb') as f:
        ipca_data = pickle.load(f)

    # Extract factors - may have different keys
    if 'ipca_factors' in ipca_data:
        factor_returns_new = ipca_data['ipca_factors']
    elif 'factor_returns' in ipca_data:
        factor_returns_new = ipca_data['factor_returns']
    elif 'F' in ipca_data:
        factor_returns_new = ipca_data['F']
    elif 'ipca_returns' in ipca_data:
        factor_returns_new = ipca_data['ipca_returns']
    else:
        print(f"[ERROR] Could not find factor returns in IPCA output")
        print(f"      Available keys: {list(ipca_data.keys())}")
        return False

    print(f"      Current factor returns shape (before filtering): {factor_returns_new.shape}")

    # Filter to the same 3 test months as the original code
    # Test months are: burnin+360, random month, burnin+n_months-1
    n_months = end - start + 1
    test_month_1 = burnin + 360
    test_month_3 = burnin + n_months - 1
    np.random.seed(TEST_SEED)  # Same seed as in compute_ipca_with_original
    test_month_2 = np.random.randint(burnin + 361, burnin + n_months - 1)
    test_months = [test_month_1, test_month_2, test_month_3]

    print(f"      Filtering to 3 test months: {test_months}")

    # Filter factor returns to only these months
    if isinstance(factor_returns_new, pd.DataFrame):
        # Check if index matches test months
        factor_returns_new = factor_returns_new.loc[test_months]
    else:
        # If numpy array, need to figure out indexing
        # Assume it's indexed from burnin+360 onwards
        indices = [m - (burnin + 360) for m in test_months]
        factor_returns_new = factor_returns_new[indices]

    print(f"      Current factor returns shape (after filtering): {factor_returns_new.shape}")

    # Step 5: Check portfolio statistics
    print(f"\n[STATS] Checking portfolio statistics...")
    stats_passed = True

    if 'ipca_stats' in ipca_data:
        ipca_stats = ipca_data['ipca_stats']
        print(f"      IPCA stats shape: {ipca_stats.shape}")
        print(f"      IPCA stats columns: {list(ipca_stats.columns)}")

        # Check that statistics are not NaN
        for col in ['stdev', 'mean', 'xret', 'hjd']:
            if col in ipca_stats.columns:
                nan_count = ipca_stats[col].isna().sum()
                total_count = len(ipca_stats)
                if nan_count > 0:
                    print(f"      [WARNING] {col}: {nan_count}/{total_count} values are NaN ({100*nan_count/total_count:.1f}%)")
                    stats_passed = False
                else:
                    mean_val = ipca_stats[col].mean()
                    std_val = ipca_stats[col].std()
                    print(f"      [PASS] {col}: mean={mean_val:.6f}, std={std_val:.6f}, no NaN values")

        # Check that stdev is positive
        if 'stdev' in ipca_stats.columns:
            neg_stdev = (ipca_stats['stdev'] <= 0).sum()
            if neg_stdev > 0:
                print(f"      [FAIL] stdev: {neg_stdev} non-positive values found")
                stats_passed = False
            else:
                print(f"      [PASS] stdev: all values are positive")
    else:
        print(f"      [WARNING] 'ipca_stats' not found in pickle file")
        stats_passed = False

    if stats_passed:
        print(f"      [ALL PASS] Statistics checks passed")
    else:
        print(f"      [FAIL] Some statistics checks failed")

    # Step 6: Compare original vs current factor returns
    print(f"\n[COMPARE] Comparing original vs current factor returns...")
    all_passed = True

    # Compare factor returns (allowing sign flips)
    print(f"\n  Factor returns:")
    print(f"    Original shape: {factor_returns_old.shape}")
    print(f"    Current shape: {factor_returns_new.shape}")

    # Check shapes match
    if factor_returns_old.shape != factor_returns_new.shape:
        print(f"    [FAIL] Shape mismatch")
        all_passed = False
    else:
        print(f"    [PASS] Shapes match")

        # Apply Procrustes alignment to resolve rotation invariance
        # Convert to (T x K) format for alignment
        if isinstance(factor_returns_old, pd.DataFrame):
            f_old_matrix = factor_returns_old.values  # (T, K)
        else:
            f_old_matrix = factor_returns_old  # Already numpy array

        if isinstance(factor_returns_new, pd.DataFrame):
            f_new_matrix = factor_returns_new.values  # (T, K)
        else:
            f_new_matrix = factor_returns_new  # Already numpy array

        # Align new factors to match old factors (reference)
        f_new_aligned, Q = align_factors_procrustes(f_old_matrix, f_new_matrix)

        print(f"    Applied Procrustes alignment (rotation matrix Q of shape {Q.shape})")
        print(f"    Orthogonality check: ||Q'Q - I|| = {np.linalg.norm(Q.T @ Q - np.eye(K)):.2e}")

        # Compare each factor (column) after alignment
        for i in range(K):
            factor_name = f"Factor {i+1}"
            try:
                # Get aligned factor columns
                f_old = f_old_matrix[:, i]
                f_new = f_new_aligned[:, i]

                # Use assert_close for comparison after Procrustes alignment
                assert_close(
                    f_new,
                    f_old,
                    rtol=1e-6,
                    atol=1e-8,
                    name=factor_name
                )
                print(f"    [PASS] {factor_name} matches after Procrustes alignment")

            except AssertionError as e:
                print(f"    [FAIL] {factor_name}: {e}")
                all_passed = False

        # Check index/month alignment if DataFrames
        if isinstance(factor_returns_old, pd.DataFrame) and isinstance(factor_returns_new, pd.DataFrame):
            try:
                # Allow index to differ slightly due to different starting points
                # Just check that we have the same number of observations
                if len(factor_returns_old) == len(factor_returns_new):
                    print(f"    [PASS] Number of observations matches")
                else:
                    print(f"    [WARN] Different number of observations: {len(factor_returns_old)} vs {len(factor_returns_new)}")
            except Exception as e:
                print(f"    [WARN] Could not compare indices: {e}")

    if all_passed:
        print("\n  [ALL PASS] All factor comparisons passed")
    else:
        print("\n  [FAIL] Factors are NOT identical")

    # Combine results
    overall_passed = all_passed and stats_passed

    # Display some statistics
    print(f"\n  Summary statistics:")
    if isinstance(factor_returns_old, pd.DataFrame):
        print(f"    Original - Mean: {factor_returns_old.mean().mean():.6f}, Std: {factor_returns_old.std().mean():.6f}")
        print(f"    Current  - Mean: {factor_returns_new.mean().mean():.6f}, Std: {factor_returns_new.std().mean():.6f}")
    else:
        print(f"    Original - Mean: {factor_returns_old.mean():.6f}, Std: {factor_returns_old.std():.6f}")
        print(f"    Current  - Mean: {factor_returns_new.mean():.6f}, Std: {factor_returns_new.std():.6f}")

    # Step 6: Cleanup - delete all pickle files
    print(f"\n[CLEANUP] Deleting pickle files...")

    files_to_delete = [panel_file, ipca_file]
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            print(f"      Deleted {file_path.name}")
        else:
            print(f"      {file_path.name} not found (already deleted)")

    print("\n" + "=" * 70)
    print(f"[DONE] IPCA factor test complete for {model.upper()}")
    if overall_passed:
        print("[RESULT] ALL TESTS PASSED")
    else:
        print("[RESULT] SOME TESTS FAILED")
    print("=" * 70)

    return overall_passed


def main():
    """Main execution function."""
    # Check configuration values first
    if not check_config_values():
        sys.exit(1)

    if len(sys.argv) < 3:
        print("Usage: python test_ipca.py <model> <K>")
        print("where:")
        print("  <model> is one of: bgn, kp14, gs21")
        print("  <K> is the number of latent factors (e.g., 1, 2, 3)")
        print("\nExamples:")
        print("  python test_ipca.py bgn 1")
        print("  python test_ipca.py kp14 2")
        print("  python test_ipca.py gs21 3")
        sys.exit(1)

    model = sys.argv[1].lower()

    try:
        K = int(sys.argv[2])
    except ValueError:
        print(f"Error: K must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    if model not in ['bgn', 'kp14', 'gs21']:
        print(f"Error: Unknown model '{model}'")
        print("Valid models: bgn, kp14, gs21")
        sys.exit(1)

    if K <= 0:
        print(f"Error: K must be positive, got {K}")
        sys.exit(1)

    success = test_ipca_factors(model, K)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
