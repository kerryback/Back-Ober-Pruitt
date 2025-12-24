"""
Test Fama-French and Fama-MacBeth Factor Computation

USAGE:
------
Run this test from the tests/ directory:
    cd tests
    python test_fama.py <model>

where <model> is one of: bgn, kp14, gs21

Directory structure (relative to tests/):
    ./original_code/        - Original Fama code to compare against
    ../run_fama.py          - Current Fama factor computation
    ../config.py            - Configuration (N, T, burnin values)

PURPOSE:
--------
This test verifies that the refactored Fama-French and Fama-MacBeth factor
computation produces numerically identical results to the original code
when given the same panel data.

WORKFLOW COMPARISON:
-------------------

ORIGINAL CODE (tests/original_code/):
  Implementation: Import directly and call functions in-process

  1. Import fama_functions.py (original Fama factor computation module)
     - Imports: parameters.py (model parameters)
     - Uses sklearn Ridge regression

  2. Functions:
     - fama_french(data, chars, mve): Compute FF portfolios (2x3 sorts)
       Returns: weights array (N, K+1) with factor weights + market

     - fama_macbeth(data, chars, mve, stdz_fm): Compute FM factors
       Returns: weights array (N, K+1) with factor loadings

     - factors(func, panel, ...): Wrapper that applies func to each month
       Returns: DataFrame of factor returns (months x factors)

  3. Execution:
     - Import fama_functions module
     - Call fama_french() and fama_macbeth() with panel data
     - Returns factor DataFrames directly

CURRENT CODE (utils_factors/):
  Implementation: Run via command line subprocess (../run_fama.py)

  1. Import fama_functions.py (refactored version)
     - Imports: config.py (centralized configuration)
     - Imports: ridge_utils.py (optimized ridge regression)
     - Imports: factor_utils.py (standardization utilities)

  2. Same function signatures as original code
     - Identical implementation with performance optimizations
     - Vectorized operations where possible
     - Cleaner, more readable code

  3. Execution:
     - Run run_fama.py via subprocess: python run_fama.py bgn_0
     - Script reads panel from pickle file (bgn_0_panel.pkl)
     - Computes FF and FM factors using fama_functions
     - Saves factors to pickle file: bgn_0_fama.pkl
     - Test reads pickle file to get results

TEST CONFIGURATION:
------------------
From tests/test_utils/config_override.py:
  - TEST_N = 50 firms
  - TEST_T = 400 time periods
  - TEST_BURNIN = 300 months
  - TEST_SEED = 12345

EXPECTED RESULTS:
----------------
All factor calculations should be numerically identical:
  1. ff_returns (Fama-French): DataFrame, rtol=1e-14, atol=1e-15
  2. fm_returns (Fama-MacBeth): DataFrame, rtol=1e-14, atol=1e-15

TEST PROCEDURE:
--------------
1. Generate panel with current code via generate_panel.py subprocess
2. Read panel pickle file
3. Compute Fama factors with original code using panel data
4. Compute Fama factors with current code via run_fama.py subprocess
5. Read fama pickle file from current code
6. Compare original vs current factor returns
7. Delete all pickle files (cleanup)

USAGE:
------
python test_fama.py <model>

where <model> is one of: bgn, kp14, gs21

Examples:
    python test_fama.py bgn
    python test_fama.py kp14
    python test_fama.py gs21
"""

import sys
import numpy as np
import pandas as pd
import pickle
import subprocess
from pathlib import Path

# Add paths (run from tests/ directory)
# ../  - for current code (config.py, generate_panel.py, etc.)
# ./test_utils/  - for test utilities (comparison.py)
# ./original_code/  - for original code to compare against
sys.path.insert(0, str(Path(__file__).parent.parent))  # ../
sys.path.insert(0, str(Path(__file__).parent / 'test_utils'))  # ./test_utils/
sys.path.insert(0, str(Path(__file__).parent / 'original_code'))  # ./original_code/

from comparison import assert_close
from config import N, T

# Test configuration
TEST_SEED = 12345


def prepare_panel_for_fama(panel, chars):
    """Prepare panel data for Fama factor computation."""
    # Set multi-index
    panel = panel.set_index(['month', 'firmid'])

    # Get valid data range
    start = panel.index.get_level_values('month').min()
    end = panel.index.get_level_values('month').max()

    return panel, start, end


def compute_fama_with_original(panel, chars, start, end, n_jobs=1):
    """Compute Fama factors using original code."""
    import fama_functions as fama_old

    print("Computing Fama factors with original code...")

    # Compute Fama-French factors
    ff_returns_old = fama_old.factors(
        fama_old.fama_french,
        panel,
        n_jobs=n_jobs,
        start=start,
        end=end,
        chars=chars
    )

    # Compute Fama-MacBeth factors
    # Note: Original code doesn't support stdz_fm parameter
    fm_returns_old = fama_old.factors(
        fama_old.fama_macbeth,
        panel,
        n_jobs=n_jobs,
        start=start,
        end=end,
        chars=chars
    )

    print(f"  Original FF returns: {ff_returns_old.shape}")
    print(f"  Original FM returns: {fm_returns_old.shape}")

    return ff_returns_old, fm_returns_old


def test_fama_factors(model):
    """Test Fama factor computation for a given model."""
    # Get burnin for this model
    from config import BGN_BURNIN, KP14_BURNIN, GS21_BURNIN
    burnin_map = {'bgn': BGN_BURNIN, 'kp14': KP14_BURNIN, 'gs21': GS21_BURNIN}
    burnin = burnin_map[model]

    print("=" * 70)
    print(f"Fama-French & Fama-MacBeth Factor Test: {model.upper()}")
    print("=" * 70)
    print(f"N={N}, T={T}, Burnin={burnin}, Seed={TEST_SEED}\n")

    # Test panel identifier
    TEST_PANEL_ID = 0
    panel_id = f"{model}_{TEST_PANEL_ID}"

    # Get model configuration
    if model == 'bgn':
        chars = ['size', 'bm', 'agr', 'roe', 'mom']
    elif model == 'kp14':
        chars = ['size', 'bm', 'agr', 'roe', 'mom']
    elif model == 'gs21':
        chars = ['size', 'bm', 'agr', 'roe', 'mom']
    else:
        raise ValueError(f"Unknown model: {model}")

    # Step 1: Generate panel with current code via command line
    print("[1/5] Generating panel with current code via generate_panel.py...")
    test_dir = Path(__file__).parent
    cmd = [sys.executable, str(test_dir / 'run_generate_panel.py'), model, str(TEST_PANEL_ID)]
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

    # Step 3: Compute Fama factors with original code
    print(f"\n[3/5] Computing Fama factors with original code...")
    panel_prepared, start, end = prepare_panel_for_fama(panel.copy(), chars)
    ff_returns_old, fm_returns_old = compute_fama_with_original(
        panel_prepared, chars, start, end
    )

    # Step 4: Compute Fama factors with current code via run_fama.py
    print(f"\n[4/5] Computing Fama factors with current code via run_fama.py...")
    parent_dir = Path(__file__).parent.parent
    cmd = [sys.executable, str(parent_dir / 'run_fama.py'), panel_id]
    result = subprocess.run(cmd, cwd=str(parent_dir), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] run_fama.py failed:")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"      Fama factors computed successfully")

    # Read fama pickle file
    fama_file = Path(DATA_DIR) / f'{panel_id}_fama.pkl'
    print(f"\n[5/5] Reading Fama factors from {fama_file}...")
    with open(fama_file, 'rb') as f:
        fama_data = pickle.load(f)
    ff_returns_new = fama_data['ff_returns']
    fm_returns_new = fama_data['fm_returns']
    print(f"      FF returns shape: {ff_returns_new.shape}")
    print(f"      FM returns shape: {fm_returns_new.shape}")

    # Step 5: Compare original vs current factor returns
    print(f"\n[COMPARE] Comparing original vs current factor returns...")
    all_passed = True

    # Compare Fama-French returns
    print(f"\n  Fama-French returns:")
    print(f"    Original columns: {list(ff_returns_old.columns)}")
    print(f"    Current columns: {list(ff_returns_new.columns)}")

    try:
        # Only compare overlapping columns
        common_cols = [col for col in ff_returns_old.columns if col in ff_returns_new.columns]
        print(f"    Comparing {len(common_cols)} common columns: {common_cols}")

        # Compare values
        assert_close(
            ff_returns_new[common_cols].values,
            ff_returns_old[common_cols].values,
            rtol=1e-14,
            atol=1e-15,
            name="ff_returns"
        )
        print(f"    [PASS] Values are identical")

        # Check index match
        assert ff_returns_new.index.equals(ff_returns_old.index), \
            "FF returns index mismatch"
        print(f"    [PASS] Index matches")

        # Note differences in columns
        extra_cols = [col for col in ff_returns_new.columns if col not in ff_returns_old.columns]
        if extra_cols:
            print(f"    [NOTE] Current code has {len(extra_cols)} extra columns: {extra_cols}")

    except AssertionError as e:
        print(f"    [FAIL] ff_returns: {e}")
        all_passed = False

    # Compare Fama-MacBeth returns
    print(f"\n  Fama-MacBeth returns:")
    print(f"    Original columns: {list(fm_returns_old.columns)}")
    print(f"    Current columns: {list(fm_returns_new.columns)}")

    try:
        # Only compare overlapping columns
        common_cols = [col for col in fm_returns_old.columns if col in fm_returns_new.columns]
        print(f"    Comparing {len(common_cols)} common columns: {common_cols}")

        # Compare values
        assert_close(
            fm_returns_new[common_cols].values,
            fm_returns_old[common_cols].values,
            rtol=1e-14,
            atol=1e-15,
            name="fm_returns"
        )
        print(f"    [PASS] Values are identical")

        # Check index match
        assert fm_returns_new.index.equals(fm_returns_old.index), \
            "FM returns index mismatch"
        print(f"    [PASS] Index matches")

        # Note differences in columns
        extra_cols = [col for col in fm_returns_new.columns if col not in fm_returns_old.columns]
        if extra_cols:
            print(f"    [NOTE] Current code has {len(extra_cols)} extra columns: {extra_cols}")

    except AssertionError as e:
        print(f"    [FAIL] fm_returns: {e}")
        all_passed = False

    if all_passed:
        print("\n  [ALL PASS] All factor comparisons passed")
    else:
        print("\n  [FAIL] Factors are NOT identical")

    # Step 6: Cleanup - delete all pickle files
    print(f"\n[CLEANUP] Deleting pickle files...")

    files_to_delete = [panel_file, fama_file]
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            print(f"      Deleted {file_path.name}")
        else:
            print(f"      {file_path.name} not found (already deleted)")

    print("\n" + "=" * 70)
    print(f"[DONE] Fama factor test complete for {model.upper()}")
    print("=" * 70)

    return all_passed


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python test_fama.py <model>")
        print("where <model> is one of: bgn, kp14, gs21")
        sys.exit(1)

    model = sys.argv[1].lower()

    if model not in ['bgn', 'kp14', 'gs21']:
        print(f"Error: Unknown model '{model}'")
        print("Valid models: bgn, kp14, gs21")
        sys.exit(1)

    success = test_fama_factors(model)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
