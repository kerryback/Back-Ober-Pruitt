"""
Master test runner for all regression tests.

Runs regression tests for all three models (BGN, KP14, GS21) and
provides a comprehensive summary of results.

Usage:
    python run_all_regression_tests.py

Or run specific models:
    python run_all_regression_tests.py bgn
    python run_all_regression_tests.py kp14 gs21
"""

import sys
import subprocess
from pathlib import Path


def run_test_file(test_file, model_name):
    """
    Run a single test file and return results.

    Parameters
    ----------
    test_file : Path
        Path to test file
    model_name : str
        Model name for display

    Returns
    -------
    tuple
        (success: bool, returncode: int)
    """
    print("=" * 80)
    print(f"RUNNING {model_name.upper()} REGRESSION TESTS")
    print("=" * 80)
    print()

    result = subprocess.run(
        [sys.executable, str(test_file)],
        cwd=test_file.parent
    )

    print()
    return (result.returncode == 0, result.returncode)


def main():
    """Main execution function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        requested_models = [arg.lower() for arg in sys.argv[1:]]
    else:
        requested_models = ['bgn', 'kp14', 'gs21']

    # Validate requested models
    valid_models = {'bgn', 'kp14', 'gs21'}
    for model in requested_models:
        if model not in valid_models:
            print(f"[ERROR] Unknown model: {model}")
            print(f"Valid models: {', '.join(sorted(valid_models))}")
            return 1

    tests_dir = Path(__file__).parent

    # Map models to test files
    test_files = {
        'bgn': tests_dir / 'test_regression_bgn.py',
        'kp14': tests_dir / 'test_regression_kp14.py',
        'gs21': tests_dir / 'test_regression_gs21.py',
    }

    # Run tests
    results = {}
    for model in requested_models:
        test_file = test_files[model]
        if not test_file.exists():
            print(f"[ERROR] Test file not found: {test_file}")
            results[model] = (False, 1)
            continue

        success, returncode = run_test_file(test_file, model)
        results[model] = (success, returncode)

    # Print summary
    print()
    print("=" * 80)
    print("REGRESSION TEST SUMMARY")
    print("=" * 80)
    print()

    all_passed = True
    for model in requested_models:
        success, returncode = results[model]
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {model.upper():<6} (exit code: {returncode})")
        if not success:
            all_passed = False

    print()
    if all_passed:
        print("[PASS] All regression tests passed!")
        print()
        print("Refactored code produces identical results to original code.")
        return 0
    else:
        failed_count = sum(1 for success, _ in results.values() if not success)
        print(f"[FAIL] {failed_count} model(s) failed regression tests")
        print()
        print("Please review test output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
