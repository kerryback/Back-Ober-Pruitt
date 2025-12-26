"""
Run All Tests - Comprehensive Test Suite Runner

This script runs all regression tests in the correct order:
1. Panel generation tests (BGN, KP14, GS21)
2. Moment calculation tests (BGN, KP14, GS21)
3. Factor computation tests (Fama, DKKM, IPCA)

Usage:
    cd tests
    python run_all_tests.py

Requirements:
    - N = 50, T = 400 (configured in config.py)
    - All dependencies installed
    - Sufficient disk space for temporary files
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for config access
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import N, T


def check_configuration():
    """Check if N and T are set to test values."""
    print("=" * 70)
    print("CONFIGURATION CHECK")
    print("=" * 70)
    print(f"Current configuration:")
    print(f"  N = {N}")
    print(f"  T = {T}")
    print()

    if N != 50 or T != 400:
        print("WARNING: Tests are designed for N=50 and T=400")
        print("Running with different values may cause tests to fail or take much longer.")
        print()
        response = input("Do you want to continue anyway? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nTests cancelled. Please update config.py to set N=50 and T=400.")
            return False
        print()

    return True


def run_test(test_file, description):
    """Run a single test file and return success status."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)

    import subprocess
    result = subprocess.run([sys.executable, test_file], cwd=str(Path(__file__).parent))

    if result.returncode != 0:
        print(f"\n[FAIL] {description} failed")
        return False
    else:
        print(f"\n[PASS] {description} passed")
        return True


def main():
    """Run all tests in sequence."""
    if not check_configuration():
        sys.exit(1)

    print("=" * 70)
    print("STARTING FULL TEST SUITE")
    print("=" * 70)
    print("\nThis will run all regression tests in the following order:")
    print("  1. Panel generation tests (BGN, KP14, GS21)")
    print("  2. Moment calculation tests (BGN, KP14, GS21)")
    print("  3. Factor computation tests (Fama, DKKM, IPCA)")
    print()

    # Track results
    results = {}

    # Phase 1: Panel generation tests
    print("\n" + "=" * 70)
    print("PHASE 1: PANEL GENERATION TESTS")
    print("=" * 70)

    panel_tests = [
        ('test_panel_bgn.py', 'BGN Panel Generation'),
        ('test_panel_kp14.py', 'KP14 Panel Generation'),
        ('test_panel_gs21.py', 'GS21 Panel Generation'),
    ]

    for test_file, description in panel_tests:
        results[description] = run_test(test_file, description)

    # Phase 2: Moment calculation tests
    print("\n" + "=" * 70)
    print("PHASE 2: MOMENT CALCULATION TESTS")
    print("=" * 70)

    moment_tests = [
        ('test_moments_bgn.py', 'BGN Moment Calculation'),
        ('test_moments_kp14.py', 'KP14 Moment Calculation'),
        ('test_moments_gs21.py', 'GS21 Moment Calculation'),
    ]

    for test_file, description in moment_tests:
        results[description] = run_test(test_file, description)

    # Phase 3: Factor computation tests
    print("\n" + "=" * 70)
    print("PHASE 3: FACTOR COMPUTATION TESTS")
    print("=" * 70)

    factor_tests = [
        ('test_fama.py', 'Fama Factor Computation'),
        ('test_dkkm.py', 'DKKM Factor Computation'),
        ('test_ipca.py', 'IPCA Factor Computation'),
    ]

    for test_file, description in factor_tests:
        results[description] = run_test(test_file, description)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    passed = sum(1 for status in results.values() if status)
    failed = sum(1 for status in results.values() if not status)

    print(f"\nTotal tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if failed > 0:
        print("Failed tests:")
        for test_name, status in results.items():
            if not status:
                print(f"  - {test_name}")

    print("\n" + "=" * 70)
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"TESTS FAILED: {failed}/{len(results)}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
