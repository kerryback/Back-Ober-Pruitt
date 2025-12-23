"""
Regression tests for GS21 model: refactored vs original code.

Verifies that refactoring preserved numerical correctness by comparing
outputs from the refactored code (current directory) against the original
code (parent directory).

Usage:
    python test_regression_gs21.py
"""

import sys
import os
import numpy as np
import subprocess
import pickle
from pathlib import Path

# Add test utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'test_utils'))
from comparison import (
    assert_close,
    assert_dataframes_equal,
    assert_factors_equal_up_to_sign,
    print_comparison_summary,
    compute_summary_stats
)
from config_override import (
    TEST_N,
    TEST_T,
    TEST_BURNIN,
    TEST_SEED,
    TEST_PANEL_ID
)

# Set paths
REFACTORED_DIR = Path(__file__).parent.parent
ORIGINAL_DIR = REFACTORED_DIR.parent


class TestGS21Regression:
    """Test suite for GS21 model regression."""

    def __init__(self):
        self.test_id = TEST_PANEL_ID
        self.panel_id = f"gs21_{self.test_id}"
        self.model = "gs21"

        # Ensure outputs directories exist
        (REFACTORED_DIR / 'outputs').mkdir(exist_ok=True)
        (ORIGINAL_DIR / 'outputs').mkdir(exist_ok=True)

        print(f"\nTest Configuration:")
        print(f"  Model: {self.model}")
        print(f"  Panel ID: {self.panel_id}")
        print(f"  N={TEST_N}, T={TEST_T}, Burnin={TEST_BURNIN}")
        print(f"  Random seed: {TEST_SEED}")
        print(f"  Refactored dir: {REFACTORED_DIR}")
        print(f"  Original dir: {ORIGINAL_DIR}")
        print()

    def test_01_panel_generation(self):
        """Test that panel generation produces identical results."""
        print("=" * 70)
        print("TEST 1: Panel Generation")
        print("=" * 70)

        # Run refactored version
        print("\n[1/2] Running refactored code...")
        result = subprocess.run(
            [sys.executable, 'generate_panel.py', self.model, str(self.test_id)],
            cwd=REFACTORED_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("[ERROR] Refactored code failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Refactored generate_panel.py failed")

        print("[OK] Refactored code completed")

        # Run original version
        print("\n[2/2] Running original code...")
        result = subprocess.run(
            [sys.executable, f'generate_panel_{self.model}.py', self.model, str(self.test_id)],
            cwd=ORIGINAL_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("[ERROR] Original code failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"Original generate_panel_{self.model}.py failed")

        print("[OK] Original code completed")

        # Load results
        print("\n[COMPARE] Loading results...")
        refactored_path = REFACTORED_DIR / 'outputs' / f'{self.panel_id}_panel.pkl'
        original_path = ORIGINAL_DIR / 'outputs' / f'{self.panel_id}_panel.pkl'

        with open(refactored_path, 'rb') as f:
            refactored = pickle.load(f)

        with open(original_path, 'rb') as f:
            original = pickle.load(f)

        print(f"  Refactored keys: {list(refactored.keys())}")
        print(f"  Original keys: {list(original.keys())}")

        # Compare panel DataFrame
        print("\n[COMPARE] Comparing panel DataFrame...")
        assert_dataframes_equal(
            refactored['panel'],
            original['panel'],
            rtol=1e-14,
            atol=1e-15,
            check_index=False  # Index might be stored differently
        )
        print("  [PASS] Panel DataFrames match")

        # Compare arrays (GS21 has 6 characteristics: A_1 through A_6)
        array_keys = [
            'A_1_taylor', 'A_2_taylor', 'A_3_taylor',
            'A_4_taylor', 'A_5_taylor', 'A_6_taylor',
            'A_1_proj', 'A_2_proj', 'A_3_proj',
            'A_4_proj', 'A_5_proj', 'A_6_proj',
            'f_1_taylor', 'f_2_taylor', 'f_3_taylor',
            'f_4_taylor', 'f_5_taylor', 'f_6_taylor',
            'f_1_proj', 'f_2_proj', 'f_3_proj',
            'f_4_proj', 'f_5_proj', 'f_6_proj'
        ]

        print("\n[COMPARE] Comparing arrays...")
        for key in array_keys:
            if key in refactored and key in original:
                assert_close(
                    refactored[key],
                    original[key],
                    rtol=1e-14,
                    atol=1e-15,
                    name=key
                )
                print(f"  [PASS] {key} matches")
            else:
                print(f"  [SKIP] {key} not in both outputs")

        # Print summary
        print("\n[SUMMARY]")
        ref_stats = compute_summary_stats(refactored, "refactored")
        orig_stats = compute_summary_stats(original, "original")
        print_comparison_summary(ref_stats, orig_stats)

        print("\n" + "=" * 70)
        print("[PASS] Panel generation test passed")
        print("=" * 70)

        return True

    def test_02_moments(self):
        """Test that moment calculation produces identical results."""
        print("=" * 70)
        print("TEST 2: Moment Calculation")
        print("=" * 70)

        # Run refactored version
        print("\n[1/2] Running refactored code...")
        result = subprocess.run(
            [sys.executable, 'calculate_moments.py', self.panel_id],
            cwd=REFACTORED_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("[ERROR] Refactored code failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Refactored calculate_moments.py failed")

        print("[OK] Refactored code completed")

        # Run original version
        print("\n[2/2] Running original code...")
        result = subprocess.run(
            [sys.executable, f'calculate_moments_{self.model}.py', self.panel_id],
            cwd=ORIGINAL_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("[ERROR] Original code failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"Original calculate_moments_{self.model}.py failed")

        print("[OK] Original code completed")

        # Load results
        print("\n[COMPARE] Loading results...")
        refactored_path = REFACTORED_DIR / 'outputs' / f'{self.panel_id}_moments.pkl'
        original_path = ORIGINAL_DIR / 'outputs' / f'{self.panel_id}_moments.pkl'

        with open(refactored_path, 'rb') as f:
            refactored = pickle.load(f)

        with open(original_path, 'rb') as f:
            original = pickle.load(f)

        print(f"  Refactored keys: {list(refactored.keys())}")
        print(f"  Original keys: {list(original.keys())}")

        # Compare moment arrays
        moment_keys = ['rp', 'cond_var', 'second_moment', 'second_moment_inv']

        print("\n[COMPARE] Comparing moment arrays...")
        for key in moment_keys:
            if key in refactored and key in original:
                # Use slightly lower tolerance for matrix inversions
                tol = 1e-10 if 'inv' in key else 1e-12
                assert_close(
                    refactored[key],
                    original[key],
                    rtol=tol,
                    atol=tol * 10,
                    name=key
                )
                print(f"  [PASS] {key} matches")
            else:
                print(f"  [SKIP] {key} not in both outputs")

        print("\n" + "=" * 70)
        print("[PASS] Moment calculation test passed")
        print("=" * 70)

        return True

    def test_03_fama(self):
        """Test that Fama factor calculation produces identical results."""
        print("=" * 70)
        print("TEST 3: Fama Factors")
        print("=" * 70)

        # Run refactored version
        print("\n[1/2] Running refactored code...")
        result = subprocess.run(
            [sys.executable, 'run_fama.py', self.panel_id],
            cwd=REFACTORED_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("[ERROR] Refactored code failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Refactored run_fama.py failed")

        print("[OK] Refactored code completed")

        # Run original version
        print("\n[2/2] Running original code...")
        result = subprocess.run(
            [sys.executable, 'run_fama.py', self.panel_id],
            cwd=ORIGINAL_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("[ERROR] Original code failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("Original run_fama.py failed")

        print("[OK] Original code completed")

        # Load results
        print("\n[COMPARE] Loading results...")
        refactored_path = REFACTORED_DIR / 'outputs' / f'{self.panel_id}_fama.pkl'
        original_path = ORIGINAL_DIR / 'outputs' / f'{self.panel_id}_fama.pkl'

        with open(refactored_path, 'rb') as f:
            refactored = pickle.load(f)

        with open(original_path, 'rb') as f:
            original = pickle.load(f)

        print(f"  Refactored keys: {list(refactored.keys())}")
        print(f"  Original keys: {list(original.keys())}")

        # Compare all factor arrays
        print("\n[COMPARE] Comparing factor arrays...")
        for key in refactored.keys():
            if key in original:
                assert_close(
                    refactored[key],
                    original[key],
                    rtol=1e-10,
                    atol=1e-12,
                    name=key
                )
                print(f"  [PASS] {key} matches")

        print("\n" + "=" * 70)
        print("[PASS] Fama factors test passed")
        print("=" * 70)

        return True

    def test_04_dkkm(self):
        """Test that DKKM factor calculation produces identical results."""
        print("=" * 70)
        print("TEST 4: DKKM Factors")
        print("=" * 70)

        from test_utils.config_override import TEST_DKKM_FEATURES

        for n_features in TEST_DKKM_FEATURES:
            print(f"\n--- Testing DKKM with {n_features} features ---")

            # Run refactored version
            print(f"\n[1/2] Running refactored code (n_features={n_features})...")
            result = subprocess.run(
                [sys.executable, 'run_dkkm.py', self.panel_id, str(n_features)],
                cwd=REFACTORED_DIR,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("[ERROR] Refactored code failed:")
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError(f"Refactored run_dkkm.py failed for n_features={n_features}")

            print("[OK] Refactored code completed")

            # Run original version
            print(f"\n[2/2] Running original code (n_features={n_features})...")
            result = subprocess.run(
                [sys.executable, 'run_dkkm.py', self.panel_id, str(n_features)],
                cwd=ORIGINAL_DIR,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("[ERROR] Original code failed:")
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError(f"Original run_dkkm.py failed for n_features={n_features}")

            print("[OK] Original code completed")

            # Load results
            print("\n[COMPARE] Loading results...")
            refactored_path = REFACTORED_DIR / 'outputs' / f'{self.panel_id}_dkkm_{n_features}.pkl'
            original_path = ORIGINAL_DIR / 'outputs' / f'{self.panel_id}_dkkm_{n_features}.pkl'

            with open(refactored_path, 'rb') as f:
                refactored = pickle.load(f)

            with open(original_path, 'rb') as f:
                original = pickle.load(f)

            print(f"  Refactored keys: {list(refactored.keys())}")
            print(f"  Original keys: {list(original.keys())}")

            # Compare all factor arrays
            print("\n[COMPARE] Comparing factor arrays...")
            for key in refactored.keys():
                if key in original:
                    assert_close(
                        refactored[key],
                        original[key],
                        rtol=1e-10,
                        atol=1e-12,
                        name=key
                    )
                    print(f"  [PASS] {key} matches")

            print(f"\n[PASS] DKKM test passed for n_features={n_features}")

        print("\n" + "=" * 70)
        print("[PASS] All DKKM tests passed")
        print("=" * 70)

        return True

    def test_05_ipca(self):
        """Test that IPCA factor calculation produces identical results."""
        print("=" * 70)
        print("TEST 5: IPCA Factors")
        print("=" * 70)

        from test_utils.config_override import TEST_IPCA_K_VALUES

        for K in TEST_IPCA_K_VALUES:
            print(f"\n--- Testing IPCA with K={K} factors ---")

            # Run refactored version
            print(f"\n[1/2] Running refactored code (K={K})...")
            result = subprocess.run(
                [sys.executable, 'run_ipca.py', self.panel_id, str(K)],
                cwd=REFACTORED_DIR,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("[ERROR] Refactored code failed:")
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError(f"Refactored run_ipca.py failed for K={K}")

            print("[OK] Refactored code completed")

            # Run original version
            print(f"\n[2/2] Running original code (K={K})...")
            result = subprocess.run(
                [sys.executable, 'run_ipca.py', self.panel_id, str(K)],
                cwd=ORIGINAL_DIR,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print("[ERROR] Original code failed:")
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError(f"Original run_ipca.py failed for K={K}")

            print("[OK] Original code completed")

            # Load results
            print("\n[COMPARE] Loading results...")
            refactored_path = REFACTORED_DIR / 'outputs' / f'{self.panel_id}_ipca_{K}.pkl'
            original_path = ORIGINAL_DIR / 'outputs' / f'{self.panel_id}_ipca_{K}.pkl'

            with open(refactored_path, 'rb') as f:
                refactored = pickle.load(f)

            with open(original_path, 'rb') as f:
                original = pickle.load(f)

            print(f"  Refactored keys: {list(refactored.keys())}")
            print(f"  Original keys: {list(original.keys())}")

            # Compare factor arrays - use sign-invariant comparison for factors
            print("\n[COMPARE] Comparing factor arrays...")
            for key in refactored.keys():
                if key not in original:
                    continue

                # Check if this is a factor array (needs sign-invariant comparison)
                if 'factors' in key.lower() or key.startswith('f_'):
                    print(f"  Testing {key} (with sign invariance)...")
                    assert_factors_equal_up_to_sign(
                        refactored[key],
                        original[key],
                        rtol=1e-6,  # IPCA uses optimization, needs more tolerance
                        atol=1e-8
                    )
                    print(f"  [PASS] {key} matches (up to sign)")
                else:
                    assert_close(
                        refactored[key],
                        original[key],
                        rtol=1e-6,  # IPCA tolerance
                        atol=1e-8,
                        name=key
                    )
                    print(f"  [PASS] {key} matches")

            print(f"\n[PASS] IPCA test passed for K={K}")

        print("\n" + "=" * 70)
        print("[PASS] All IPCA tests passed")
        print("=" * 70)

        return True


def main():
    """Run all regression tests for GS21 model."""
    print("\nGS21 Model Regression Test Suite")
    print("=" * 70)

    tester = TestGS21Regression()

    tests = [
        ("Panel Generation", tester.test_01_panel_generation),
        ("Moment Calculation", tester.test_02_moments),
        ("Fama Factors", tester.test_03_fama),
        ("DKKM Factors", tester.test_04_dkkm),
        ("IPCA Factors", tester.test_05_ipca),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test_name} failed:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[PASS] All tests passed!")
        return 0
    else:
        print(f"\n[FAIL] {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
