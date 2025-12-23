"""
Regression tests comparing core functions: refactored vs original code.

This test compares the numerical output of key functions between:
- Refactored code: utils_bgn/, utils_kp14/, utils_gs21/, utils_factors/
- Original code: panel_functions_*.py, sdf_compute_*.py, *_functions.py

Usage:
    python test_regression_functions.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path to access original code
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add test utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'test_utils'))
from comparison import assert_close
from config_override import TEST_N, TEST_T, TEST_BURNIN, TEST_SEED


class TestBGNFunctions:
    """Test BGN model function equivalence."""

    def __init__(self):
        # Import refactored code
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils_bgn import panel_functions_bgn as bgn_new
        from utils_bgn import sdf_compute_bgn as sdf_new

        # Import original code
        import panel_functions as bgn_old
        import sdf_compute as sdf_old

        self.bgn_new = bgn_new
        self.bgn_old = bgn_old
        self.sdf_new = sdf_new
        self.sdf_old = sdf_old

        print("\nTest Configuration:")
        print(f"  Model: BGN")
        print(f"  N={TEST_N}, T={TEST_T}, Burnin={TEST_BURNIN}")
        print(f"  Random seed: {TEST_SEED}")
        print()

    def test_01_create_arrays(self):
        """Test that create_arrays produces identical results."""
        print("=" * 70)
        print("TEST 1: BGN create_arrays()")
        print("=" * 70)

        np.random.seed(TEST_SEED)

        # Run refactored version
        print("\n[1/2] Running refactored code...")
        arrays_new = self.bgn_new.create_arrays(TEST_N, TEST_T + TEST_BURNIN)
        print("[OK] Refactored code completed")

        # Run original version
        print("\n[2/2] Running original code...")
        np.random.seed(TEST_SEED)  # Reset seed for identical randomness
        arrays_old = self.bgn_old.create_arrays(TEST_N, TEST_T + TEST_BURNIN)
        print("[OK] Original code completed")

        # Compare arrays
        print("\n[COMPARE] Comparing arrays...")
        array_names = ['A_1_taylor', 'A_2_taylor', 'A_1_proj', 'A_2_proj',
                       'f_1_taylor', 'f_2_taylor', 'f_1_proj', 'f_2_proj']

        for i, name in enumerate(array_names):
            assert_close(
                arrays_new[i],
                arrays_old[i],
                rtol=1e-14,
                atol=1e-15,
                name=name
            )
            print(f"  [PASS] {name} matches")

        print("\n" + "=" * 70)
        print("[PASS] create_arrays test passed")
        print("=" * 70)

        return arrays_new, arrays_old

    def test_02_create_panel(self):
        """Test that create_panel produces identical DataFrames."""
        print("\n" + "=" * 70)
        print("TEST 2: BGN create_panel()")
        print("=" * 70)

        np.random.seed(TEST_SEED)
        arrays_new = self.bgn_new.create_arrays(TEST_N, TEST_T + TEST_BURNIN)

        np.random.seed(TEST_SEED)
        arrays_old = self.bgn_old.create_arrays(TEST_N, TEST_T + TEST_BURNIN)

        # Run refactored version
        print("\n[1/2] Running refactored code...")
        panel_new = self.bgn_new.create_panel(TEST_N, TEST_T + TEST_BURNIN, arrays_new)
        print("[OK] Refactored code completed")

        # Run original version
        print("\n[2/2] Running original code...")
        panel_old = self.bgn_old.create_panel(TEST_N, TEST_T + TEST_BURNIN, arrays_old)
        print("[OK] Original code completed")

        # Compare panels
        print("\n[COMPARE] Comparing panels...")
        print(f"  Refactored shape: {panel_new.shape}")
        print(f"  Original shape: {panel_old.shape}")

        # Compare columns
        for col in panel_new.columns:
            if col in panel_old.columns:
                assert_close(
                    panel_new[col].values,
                    panel_old[col].values,
                    rtol=1e-14,
                    atol=1e-15,
                    name=f"column '{col}'"
                )
                print(f"  [PASS] Column '{col}' matches")

        print("\n" + "=" * 70)
        print("[PASS] create_panel test passed")
        print("=" * 70)

        return panel_new, panel_old


class TestKP14Functions:
    """Test KP14 model function equivalence."""

    def __init__(self):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils_kp14 import panel_functions_kp14 as kp_new
        from utils_kp14 import sdf_compute_kp14 as sdf_new

        import panel_functions_kp14 as kp_old
        import sdf_compute_kp14 as sdf_old

        self.kp_new = kp_new
        self.kp_old = kp_old
        self.sdf_new = sdf_new
        self.sdf_old = sdf_old

        print("\nTest Configuration:")
        print(f"  Model: KP14")
        print(f"  N={TEST_N}, T={TEST_T}, Burnin={TEST_BURNIN}")
        print(f"  Random seed: {TEST_SEED}")
        print()

    def test_01_create_arrays(self):
        """Test that create_arrays produces identical results."""
        print("=" * 70)
        print("TEST 1: KP14 create_arrays()")
        print("=" * 70)

        np.random.seed(TEST_SEED)

        print("\n[1/2] Running refactored code...")
        arrays_new = self.kp_new.create_arrays(TEST_N, TEST_T + TEST_BURNIN)
        print("[OK] Refactored code completed")

        print("\n[2/2] Running original code...")
        np.random.seed(TEST_SEED)
        arrays_old = self.kp_old.create_arrays(TEST_N, TEST_T + TEST_BURNIN)
        print("[OK] Original code completed")

        print("\n[COMPARE] Comparing arrays...")
        array_names = ['A_taylor', 'A_proj', 'f_taylor', 'f_proj']

        for i, name in enumerate(array_names):
            assert_close(
                arrays_new[i],
                arrays_old[i],
                rtol=1e-14,
                atol=1e-15,
                name=name
            )
            print(f"  [PASS] {name} matches")

        print("\n" + "=" * 70)
        print("[PASS] create_arrays test passed")
        print("=" * 70)

        return True


class TestGS21Functions:
    """Test GS21 model function equivalence."""

    def __init__(self):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils_gs21 import panel_functions_gs21 as gs_new
        from utils_gs21 import sdf_compute_gs21 as sdf_new

        import panel_functions_gs21 as gs_old
        import sdf_compute_gs21 as sdf_old

        self.gs_new = gs_new
        self.gs_old = gs_old
        self.sdf_new = sdf_new
        self.sdf_old = sdf_old

        print("\nTest Configuration:")
        print(f"  Model: GS21")
        print(f"  N={TEST_N}, T={TEST_T}, Burnin={TEST_BURNIN}")
        print(f"  Random seed: {TEST_SEED}")
        print()

    def test_01_create_arrays(self):
        """Test that create_arrays produces identical results."""
        print("=" * 70)
        print("TEST 1: GS21 create_arrays()")
        print("=" * 70)

        np.random.seed(TEST_SEED)

        print("\n[1/2] Running refactored code...")
        arrays_new = self.gs_new.create_arrays(TEST_N, TEST_T + TEST_BURNIN)
        print("[OK] Refactored code completed")

        print("\n[2/2] Running original code...")
        np.random.seed(TEST_SEED)
        arrays_old = self.gs_old.create_arrays(TEST_N, TEST_T + TEST_BURNIN)
        print("[OK] Original code completed")

        print("\n[COMPARE] Comparing arrays...")
        # GS21 has 6 characteristics
        array_names = ['A_1_taylor', 'A_2_taylor', 'A_3_taylor',
                       'A_4_taylor', 'A_5_taylor', 'A_6_taylor',
                       'A_1_proj', 'A_2_proj', 'A_3_proj',
                       'A_4_proj', 'A_5_proj', 'A_6_proj',
                       'f_1_taylor', 'f_2_taylor', 'f_3_taylor',
                       'f_4_taylor', 'f_5_taylor', 'f_6_taylor',
                       'f_1_proj', 'f_2_proj', 'f_3_proj',
                       'f_4_proj', 'f_5_proj', 'f_6_proj']

        for i, name in enumerate(array_names):
            assert_close(
                arrays_new[i],
                arrays_old[i],
                rtol=1e-14,
                atol=1e-15,
                name=name
            )
            print(f"  [PASS] {name} matches")

        print("\n" + "=" * 70)
        print("[PASS] create_arrays test passed")
        print("=" * 70)

        return True


def main():
    """Run all function regression tests."""
    print("\nFunction Regression Test Suite")
    print("=" * 70)
    print("Testing equivalence of refactored vs original functions")
    print("=" * 70)

    all_tests = [
        ("BGN create_arrays", lambda: TestBGNFunctions().test_01_create_arrays()),
        ("BGN create_panel", lambda: TestBGNFunctions().test_02_create_panel()),
        ("KP14 create_arrays", lambda: TestKP14Functions().test_01_create_arrays()),
        ("GS21 create_arrays", lambda: TestGS21Functions().test_01_create_arrays()),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in all_tests:
        try:
            print(f"\n\nRunning: {test_name}")
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
    print(f"Tests passed: {passed}/{len(all_tests)}")
    print(f"Tests failed: {failed}/{len(all_tests)}")

    if failed == 0:
        print("\n[PASS] All function tests passed!")
        print("Refactored functions produce identical results to original functions.")
        return 0
    else:
        print(f"\n[FAIL] {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
