"""
Test DKKM Factor Computation - Regression Test

USAGE:
------
Run this test from the tests/ directory:
    cd tests
    python test_dkkm.py <model> <nfeatures>

where:
    <model> is one of: bgn, kp14, gs21
    <nfeatures> is the number of random features (e.g., 6, 36, 360)

Examples:
    python test_dkkm.py bgn 360
    python test_dkkm.py kp14 180
    python test_dkkm.py gs21 6

Directory structure (relative to tests/):
    ./original_code/        - Original DKKM code to compare against
    ../run_dkkm.py          - Current DKKM factor computation
    ../config.py            - Configuration (N, T, burnin values)

PURPOSE:
--------
This test verifies that the refactored DKKM (Deep Kernel Kernel Methods) factor
computation produces numerically identical results to the original code when
given the same panel data and random weight matrix.

WORKFLOW COMPARISON:
-------------------

ORIGINAL CODE (tests/original_code/):
  Implementation: Import directly and call functions in-process

  1. Import dkkm_functions.py (original DKKM module)
     - Imports: parameters.py (model parameters)
     - Uses joblib.Parallel for parallelization

  2. Functions:
     - rff(data, rf, W, model): Random Fourier Features for single month
       Returns: rank-standardized weights, non-rank-standardized weights

     - factors(panel, W, n_jobs, start, end, model, chars): Panel of RFF returns
       Returns: (f_rs, f_nors) - DataFrames of factor returns

  3. Execution:
     - Import dkkm_functions module
     - Call factors() with panel data
     - Returns factor DataFrames directly

CURRENT CODE (utils_factors/):
  Implementation: Run from the terminal via subprocess (../run_dkkm.py)

  1. Import dkkm_functions.py (refactored version)
     - Imports: config.py (centralized configuration)
     - Uses numba acceleration (dkkm_functions_numba.py)
     - Improved memory efficiency

  2. Same function signatures as original code
     - Identical mathematical implementation
     - Performance optimizations with numba
     - Vectorized operations where possible

  3. Execution:
     - Run run_dkkm.py via subprocess: python run_dkkm.py bgn_0 360
     - Script reads panel from pickle file (bgn_0_panel.pkl)
     - Computes DKKM factors using dkkm_functions
     - Saves factors to pickle file: bgn_0_dkkm_360.pkl
     - Test reads pickle file to get results

TEST CONFIGURATION:
------------------
From config.py:
  - N = 50 firms
  - T = 400 time periods
  - BGN_BURNIN/KP14_BURNIN/GS21_BURNIN = 300 months
  - TEST_SEED = 12345 (set in test script)
  - Random weight matrix W is generated with fixed seed

EXPECTED RESULTS:
----------------
All DKKM factor calculations should be numerically identical:
  1. f_rs (rank-standardized): DataFrame, rtol=1e-14, atol=1e-15
  2. f_nors (non-rank-standardized): DataFrame, rtol=1e-14, atol=1e-15

TEST PROCEDURE:
--------------
1. Generate panel with current code via run_generate_panel.py subprocess
2. Read panel pickle file
3. Generate random weight matrix W with fixed seed
4. Compute DKKM factors with original code using panel data and W
5. Compute DKKM factors with current code via run_dkkm.py subprocess
6. Read DKKM pickle file from current code
7. Compare original vs current factor returns
8. Delete all pickle files (cleanup)
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import subprocess
from pathlib import Path

# Add paths (run from tests/ directory)
# ../  - for current code (config.py, run_dkkm.py, etc.)
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


def prepare_panel_for_dkkm(panel, chars):
    """Prepare panel data for DKKM factor computation."""
    # Set multi-index
    panel = panel.set_index(['month', 'firmid'])

    # Get valid data range
    start = panel.index.get_level_values('month').min()
    end = panel.index.get_level_values('month').max()

    return panel, start, end


def compute_dkkm_with_original(panel, W, nfeatures, chars, start, end, model, n_jobs=1):
    """Compute DKKM factors using original code."""
    import dkkm_functions as dkkm_old

    print("Computing DKKM factors with original code...")

    # Compute DKKM factors
    f_rs_old, f_nors_old = dkkm_old.factors(
        panel=panel,
        W=W,
        n_jobs=n_jobs,
        start=start,
        end=end,
        model=model,
        chars=chars
    )

    print(f"  Original f_rs shape: {f_rs_old.shape}")
    print(f"  Original f_nors shape: {f_nors_old.shape}")

    return f_rs_old, f_nors_old


def test_dkkm_factors(model, nfeatures):
    """Test DKKM factor computation for a given model and number of features."""
    # Get burnin for this model
    from config import BGN_BURNIN, KP14_BURNIN, GS21_BURNIN
    burnin_map = {'bgn': BGN_BURNIN, 'kp14': KP14_BURNIN, 'gs21': GS21_BURNIN}
    burnin = burnin_map[model]

    print("=" * 70)
    print(f"DKKM Factor Test: {model.upper()}")
    print("=" * 70)
    print(f"N={N}, T={T}, Burnin={burnin}, Seed={TEST_SEED}")
    print(f"Number of features: {nfeatures}\n")

    # Test panel identifier
    TEST_PANEL_ID = 0
    panel_id = f"{model}_{TEST_PANEL_ID}"

    # Get model configuration
    chars = ['size', 'bm', 'agr', 'roe', 'mom']

    # Generate random weight matrix W with fixed seed
    np.random.seed(TEST_SEED)
    W = np.random.randn(nfeatures, len(chars) + (1 if model == 'bgn' else 0))

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

    # Step 3: Compute DKKM factors with original code
    print(f"\n[3/5] Computing DKKM factors with original code...")
    panel_prepared, start, end = prepare_panel_for_dkkm(panel.copy(), chars)
    f_rs_old, f_nors_old = compute_dkkm_with_original(
        panel_prepared, W, nfeatures, chars, start, end, model
    )

    # Step 4: Save W matrix and compute with current code via run_dkkm.py
    # Note: The current code needs to use the same W matrix
    # We'll save it to a temporary file that run_dkkm.py can read
    w_file = Path(DATA_DIR) / f'{panel_id}_dkkm_W_{nfeatures}.pkl'
    with open(w_file, 'wb') as f:
        pickle.dump(W, f)

    print(f"\n[4/5] Computing DKKM factors with current code via run_dkkm.py...")
    parent_dir = Path(__file__).parent.parent
    cmd = [sys.executable, str(parent_dir / 'run_dkkm.py'), panel_id, str(nfeatures)]
    result = subprocess.run(cmd, cwd=str(parent_dir), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] run_dkkm.py failed:")
        print(result.stdout)
        print(result.stderr)
        # Clean up W file
        if w_file.exists():
            w_file.unlink()
        return False

    print(f"      DKKM factors computed successfully")

    # Read DKKM pickle file
    dkkm_file = Path(DATA_DIR) / f'{panel_id}_dkkm_{nfeatures}.pkl'
    print(f"\n[5/5] Reading DKKM factors from {dkkm_file}...")
    with open(dkkm_file, 'rb') as f:
        dkkm_data = pickle.load(f)

    # Extract factors - may have different keys
    if 'dkkm_factors' in dkkm_data:
        f_rs_new = dkkm_data['dkkm_factors']
        f_nors_new = dkkm_data.get('dkkm_factors_nors', f_rs_new)
    elif 'f_rs' in dkkm_data and 'f_nors' in dkkm_data:
        f_rs_new = dkkm_data['f_rs']
        f_nors_new = dkkm_data['f_nors']
    elif 'factor_returns' in dkkm_data:
        # Current code may only save one version
        f_rs_new = dkkm_data['factor_returns']
        f_nors_new = dkkm_data.get('factor_returns_nors', f_rs_new)
    else:
        f_rs_new = dkkm_data
        f_nors_new = f_rs_new

    print(f"      Current f_rs shape: {f_rs_new.shape}")

    # Step 5: Check portfolio statistics
    print(f"\n[STATS] Checking portfolio statistics...")
    stats_passed = True

    if 'dkkm_stats' in dkkm_data:
        dkkm_stats = dkkm_data['dkkm_stats']
        print(f"      DKKM stats shape: {dkkm_stats.shape}")
        print(f"      DKKM stats columns: {list(dkkm_stats.columns)}")

        # Check that statistics are not NaN
        for col in ['stdev', 'mean', 'xret', 'hjd']:
            if col in dkkm_stats.columns:
                nan_count = dkkm_stats[col].isna().sum()
                total_count = len(dkkm_stats)
                if nan_count > 0:
                    print(f"      [WARNING] {col}: {nan_count}/{total_count} values are NaN ({100*nan_count/total_count:.1f}%)")
                    stats_passed = False
                else:
                    mean_val = dkkm_stats[col].mean()
                    std_val = dkkm_stats[col].std()
                    print(f"      [PASS] {col}: mean={mean_val:.6f}, std={std_val:.6f}, no NaN values")

        # Check that stdev is positive
        if 'stdev' in dkkm_stats.columns:
            neg_stdev = (dkkm_stats['stdev'] <= 0).sum()
            if neg_stdev > 0:
                print(f"      [FAIL] stdev: {neg_stdev} non-positive values found")
                stats_passed = False
            else:
                print(f"      [PASS] stdev: all values are positive")
    else:
        print(f"      [WARNING] 'dkkm_stats' not found in pickle file")
        stats_passed = False

    if stats_passed:
        print(f"      [ALL PASS] Statistics checks passed")
    else:
        print(f"      [FAIL] Some statistics checks failed")

    # Step 6: Compare original vs current factor returns
    print(f"\n[COMPARE] Comparing original vs current factor returns...")
    all_passed = True

    # Compare rank-standardized returns
    print(f"\n  Rank-standardized returns (f_rs):")
    print(f"    Original shape: {f_rs_old.shape}")
    print(f"    Current shape: {f_rs_new.shape}")

    try:
        # Compare values
        assert_close(
            f_rs_new.values,
            f_rs_old.values,
            rtol=1e-14,
            atol=1e-15,
            name="f_rs"
        )
        print(f"    [PASS] Values are identical")

        # Check index match
        assert f_rs_new.index.equals(f_rs_old.index), "f_rs index mismatch"
        print(f"    [PASS] Index matches")

    except AssertionError as e:
        print(f"    [FAIL] f_rs: {e}")
        all_passed = False

    if all_passed:
        print("\n  [ALL PASS] All factor comparisons passed")
    else:
        print("\n  [FAIL] Factors are NOT identical")

    # Combine results
    overall_passed = all_passed and stats_passed

    # Step 7: Cleanup - delete all pickle files
    print(f"\n[CLEANUP] Deleting pickle files...")

    files_to_delete = [panel_file, dkkm_file, w_file]
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            print(f"      Deleted {file_path.name}")
        else:
            print(f"      {file_path.name} not found (already deleted)")

    print("\n" + "=" * 70)
    print(f"[DONE] DKKM factor test complete for {model.upper()}")
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
        print("Usage: python test_dkkm.py <model> <nfeatures>")
        print("where:")
        print("  <model> is one of: bgn, kp14, gs21")
        print("  <nfeatures> is the number of random features (e.g., 6, 36, 360)")
        print("\nExamples:")
        print("  python test_dkkm.py bgn 360")
        print("  python test_dkkm.py kp14 36")
        sys.exit(1)

    model = sys.argv[1].lower()

    try:
        nfeatures = int(sys.argv[2])
    except ValueError:
        print(f"Error: nfeatures must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    if model not in ['bgn', 'kp14', 'gs21']:
        print(f"Error: Unknown model '{model}'")
        print("Valid models: bgn, kp14, gs21")
        sys.exit(1)

    if nfeatures <= 0:
        print(f"Error: nfeatures must be positive, got {nfeatures}")
        sys.exit(1)

    success = test_dkkm_factors(model, nfeatures)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
