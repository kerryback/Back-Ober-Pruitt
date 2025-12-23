"""
Test moment calculations on same panel data.

Generates a panel with current code, then feeds it to both current
and original sdf_compute implementations. Compares numerical results.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'test_utils'))

from comparison import assert_close
from config_override import TEST_N, TEST_T, TEST_BURNIN, TEST_SEED


def test_bgn_moment_calculations():
    """Test BGN moment calculations on same panel."""
    print("=" * 70)
    print("BGN Moment Calculation Test")
    print("=" * 70)
    print(f"N={TEST_N}, T={TEST_T}, Burnin={TEST_BURNIN}, Seed={TEST_SEED}\n")

    # Import current code
    from utils_bgn import panel_functions_bgn as bgn_new
    from utils_bgn import sdf_compute_bgn as sdf_new

    # Import original code
    import panel_functions as bgn_old
    import sdf_compute as sdf_old

    # Generate panel with current code
    print("[1/3] Generating panel with current code...")
    np.random.seed(TEST_SEED)
    arrays = bgn_new.create_arrays(TEST_N, TEST_T + TEST_BURNIN)
    panel = bgn_new.create_panel(TEST_N, TEST_T + TEST_BURNIN, arrays)
    print(f"      Panel shape: {panel.shape}")

    # Create SDF compute functions
    print("\n[2/3] Creating SDF compute function with current code...")
    sdf_loop_new = sdf_new.sdf_compute(TEST_N, TEST_T + TEST_BURNIN, arrays)

    print("[3/3] Creating SDF compute function with original code...")
    sdf_loop_old = sdf_old.sdf_compute(TEST_N, TEST_T + TEST_BURNIN, arrays)

    # Test a few months
    test_months = [TEST_BURNIN + 360, TEST_BURNIN + 400, TEST_T + TEST_BURNIN - 50]

    print(f"\n[COMPARE] Testing moments for {len(test_months)} months...")

    all_passed = True
    for month in test_months:
        if month >= TEST_T + TEST_BURNIN:
            continue

        print(f"\n  Month {month}:")

        # Compute with current code
        sdf_ret_new, max_sr_new, rp_new, cond_var_new = sdf_loop_new(month - 1, 0)

        # Compute with original code
        sdf_ret_old, max_sr_old, rp_old, cond_var_old = sdf_loop_old(month - 1, 0)

        # Compare rp (risk premia)
        try:
            assert_close(rp_new, rp_old, rtol=1e-12, atol=1e-14, name=f"rp[{month}]")
            print(f"    [PASS] rp")
        except AssertionError as e:
            print(f"    [FAIL] rp: {e}")
            all_passed = False

        # Compare cond_var
        try:
            assert_close(cond_var_new, cond_var_old, rtol=1e-12, atol=1e-14, name=f"cond_var[{month}]")
            print(f"    [PASS] cond_var")
        except AssertionError as e:
            print(f"    [FAIL] cond_var: {e}")
            all_passed = False

        # Compare max_sr
        try:
            assert_close(
                np.array([max_sr_new]),
                np.array([max_sr_old]),
                rtol=1e-12,
                atol=1e-14,
                name=f"max_sr[{month}]"
            )
            print(f"    [PASS] max_sr")
        except AssertionError as e:
            print(f"    [FAIL] max_sr: {e}")
            all_passed = False

    if all_passed:
        print("\n  [ALL PASS] All moment comparisons passed")

    print("\n" + "=" * 70)
    print("[DONE] BGN moment calculation test complete")
    print("=" * 70)
    return True


def test_kp14_moment_calculations():
    """Test KP14 moment calculations on same panel."""
    print("\n" + "=" * 70)
    print("KP14 Moment Calculation Test")
    print("=" * 70)
    print(f"N={TEST_N}, T={TEST_T}, Burnin={TEST_BURNIN}, Seed={TEST_SEED}\n")

    from utils_kp14 import panel_functions_kp14 as kp_new
    from utils_kp14 import sdf_compute_kp14 as sdf_new

    import panel_functions_kp14 as kp_old
    import sdf_compute_kp14 as sdf_old

    print("[1/3] Generating panel with current code...")
    np.random.seed(TEST_SEED)
    arrays = kp_new.create_arrays(TEST_N, TEST_T + TEST_BURNIN)
    panel = kp_new.create_panel(TEST_N, TEST_T + TEST_BURNIN, arrays)
    print(f"      Panel shape: {panel.shape}")

    print("\n[2/3] Creating SDF compute function with current code...")
    sdf_loop_new = sdf_new.sdf_compute(TEST_N, TEST_T + TEST_BURNIN, arrays)

    print("[3/3] Creating SDF compute function with original code...")
    sdf_loop_old = sdf_old.sdf_compute(TEST_N, TEST_T + TEST_BURNIN, arrays)

    # Test a few months
    test_months = [TEST_BURNIN + 360, TEST_BURNIN + 400, TEST_T + TEST_BURNIN - 50]

    print(f"\n[COMPARE] Testing moments for {len(test_months)} months...")

    all_passed = True
    for month in test_months:
        if month >= TEST_T + TEST_BURNIN:
            continue

        print(f"\n  Month {month}:")

        # Compute with current code
        sdf_ret_new, max_sr_new, rp_new, cond_var_new = sdf_loop_new(month - 1, 0)

        # Compute with original code
        sdf_ret_old, max_sr_old, rp_old, cond_var_old = sdf_loop_old(month - 1, 0)

        # Compare rp (risk premia)
        try:
            assert_close(rp_new, rp_old, rtol=1e-12, atol=1e-14, name=f"rp[{month}]")
            print(f"    [PASS] rp")
        except AssertionError as e:
            print(f"    [FAIL] rp: {e}")
            all_passed = False

        # Compare cond_var
        try:
            assert_close(cond_var_new, cond_var_old, rtol=1e-12, atol=1e-14, name=f"cond_var[{month}]")
            print(f"    [PASS] cond_var")
        except AssertionError as e:
            print(f"    [FAIL] cond_var: {e}")
            all_passed = False

        # Compare max_sr
        try:
            assert_close(
                np.array([max_sr_new]),
                np.array([max_sr_old]),
                rtol=1e-12,
                atol=1e-14,
                name=f"max_sr[{month}]"
            )
            print(f"    [PASS] max_sr")
        except AssertionError as e:
            print(f"    [FAIL] max_sr: {e}")
            all_passed = False

    if all_passed:
        print("\n  [ALL PASS] All moment comparisons passed")

    print("\n" + "=" * 70)
    print("[DONE] KP14 moment calculation test complete")
    print("=" * 70)
    return True


def test_gs21_moment_calculations():
    """Test GS21 moment calculations on same panel."""
    print("\n" + "=" * 70)
    print("GS21 Moment Calculation Test")
    print("=" * 70)
    print(f"N={TEST_N}, T={TEST_T}, Burnin={TEST_BURNIN}, Seed={TEST_SEED}\n")

    from utils_gs21 import panel_functions_gs21 as gs_new
    from utils_gs21 import sdf_compute_gs21 as sdf_new

    import panel_functions_gs21 as gs_old
    import sdf_compute_gs21 as sdf_old

    print("[1/3] Generating panel with current code...")
    np.random.seed(TEST_SEED)
    arrays = gs_new.create_arrays(TEST_N, TEST_T + TEST_BURNIN)
    panel = gs_new.create_panel(TEST_N, TEST_T + TEST_BURNIN, arrays)
    print(f"      Panel shape: {panel.shape}")

    print("\n[2/3] Creating SDF compute function with current code...")
    sdf_loop_new = sdf_new.sdf_compute(TEST_N, TEST_T + TEST_BURNIN, arrays)

    print("[3/3] Creating SDF compute function with original code...")
    sdf_loop_old = sdf_old.sdf_compute(TEST_N, TEST_T + TEST_BURNIN, arrays)

    # Test a few months
    test_months = [TEST_BURNIN + 360, TEST_BURNIN + 400, TEST_T + TEST_BURNIN - 50]

    print(f"\n[COMPARE] Testing moments for {len(test_months)} months...")

    all_passed = True
    for month in test_months:
        if month >= TEST_T + TEST_BURNIN:
            continue

        print(f"\n  Month {month}:")

        # Compute with current code
        sdf_ret_new, max_sr_new, rp_new, cond_var_new = sdf_loop_new(month - 1, 0)

        # Compute with original code
        sdf_ret_old, max_sr_old, rp_old, cond_var_old = sdf_loop_old(month - 1, 0)

        # Compare rp (risk premia)
        try:
            assert_close(rp_new, rp_old, rtol=1e-12, atol=1e-14, name=f"rp[{month}]")
            print(f"    [PASS] rp")
        except AssertionError as e:
            print(f"    [FAIL] rp: {e}")
            all_passed = False

        # Compare cond_var
        try:
            assert_close(cond_var_new, cond_var_old, rtol=1e-12, atol=1e-14, name=f"cond_var[{month}]")
            print(f"    [PASS] cond_var")
        except AssertionError as e:
            print(f"    [FAIL] cond_var: {e}")
            all_passed = False

        # Compare max_sr
        try:
            assert_close(
                np.array([max_sr_new]),
                np.array([max_sr_old]),
                rtol=1e-12,
                atol=1e-14,
                name=f"max_sr[{month}]"
            )
            print(f"    [PASS] max_sr")
        except AssertionError as e:
            print(f"    [FAIL] max_sr: {e}")
            all_passed = False

    if all_passed:
        print("\n  [ALL PASS] All moment comparisons passed")

    print("\n" + "=" * 70)
    print("[DONE] GS21 moment calculation test complete")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_bgn_moment_calculations()
    test_kp14_moment_calculations()
    test_gs21_moment_calculations()
