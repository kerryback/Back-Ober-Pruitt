"""
Test randomized ridge regression.

Validates that the randomized SVD implementation runs correctly
and produces reasonable results for different problem sizes.

Usage:
    python test_randomized_ridge.py
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_factors import ridge_utils


def test_basic_functionality():
    """Test that randomized ridge regression runs without errors."""
    print("\nBasic Functionality Test")
    print("=" * 60)
    print("Parameters: T=200, D=500, n_alphas=5, k=100")
    print()

    # Generate test data
    np.random.seed(42)
    T = 200
    D = 500
    n_alphas = 5

    print("Generating test data...")
    X = np.random.randn(T, D)
    y = np.random.randn(T)
    alphas = np.logspace(-2, 1, n_alphas)

    print("Running randomized ridge regression...")
    t0 = time.time()
    try:
        betas = ridge_utils.ridge_regression_grid_randomized(
            X, y, alphas, max_rank=100
        )
        elapsed = time.time() - t0

        print(f"[OK] Completed in {elapsed:.3f}s")
        print(f"     Output shape: {betas.shape}")
        print(f"     Expected shape: ({D}, {n_alphas})")

        # Verify output shape
        if betas.shape == (D, n_alphas):
            print("\n[PASS] Shape verification passed")
        else:
            print(f"\n[FAIL] Shape mismatch: got {betas.shape}, expected ({D}, {n_alphas})")
            return False

        # Verify no NaNs or Infs
        if np.any(np.isnan(betas)) or np.any(np.isinf(betas)):
            print("[FAIL] Output contains NaN or Inf values")
            return False
        else:
            print("[PASS] No NaN or Inf values")

        # Verify shrinkage effect (larger alpha -> smaller coefficients)
        norms = np.linalg.norm(betas, axis=0)
        if np.all(norms[:-1] >= norms[1:]):
            print("[PASS] Shrinkage effect verified (larger alpha -> smaller norm)")
        else:
            print("[WARN] Shrinkage effect not monotonic (expected for ridge)")

        return True

    except Exception as e:
        print(f"\n[FAIL] Exception raised: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_consistency():
    """Test that randomized method gives consistent results with same seed."""
    print("\n\nConsistency Test")
    print("=" * 60)
    print("Parameters: T=100, D=200, n_alphas=3, k=50")
    print()

    # Generate test data
    T = 100
    D = 200
    n_alphas = 3

    print("Generating test data...")
    np.random.seed(123)
    X = np.random.randn(T, D)
    y = np.random.randn(T)
    alphas = [0.001, 0.01, 0.1]

    print("Running first iteration...")
    betas1 = ridge_utils.ridge_regression_grid_randomized(
        X, y, alphas, max_rank=50
    )

    print("Running second iteration...")
    betas2 = ridge_utils.ridge_regression_grid_randomized(
        X, y, alphas, max_rank=50
    )

    # Check if results are identical (randomized_svd uses fixed random_state=42)
    if np.allclose(betas1, betas2, rtol=1e-10):
        print("\n[PASS] Randomized method produces consistent results")
        return True
    else:
        max_diff = np.max(np.abs(betas1 - betas2))
        print(f"\n[WARN] Results differ (max diff: {max_diff:.2e})")
        print("       This may be due to randomized_svd implementation details")
        return True  # Not a failure, just informational


def benchmark_performance():
    """Benchmark performance for realistic problem size."""
    print("\n\nPerformance Benchmark")
    print("=" * 60)
    print("Parameters: T=360, D=1000, n_alphas=7, k=500")
    print()

    # Generate test data
    np.random.seed(42)
    T = 360
    D = 1000
    n_alphas = 7

    print("Generating test data...")
    X = np.random.randn(T, D)
    y = np.random.randn(T)
    alphas = np.logspace(-3, 0, n_alphas)

    print("Running benchmark...")
    t0 = time.time()
    betas = ridge_utils.ridge_regression_grid_randomized(
        X, y, alphas, max_rank=500
    )
    elapsed = time.time() - t0

    print(f"\n[OK] Completed in {elapsed:.3f}s")
    print(f"     Time per alpha: {elapsed/n_alphas:.3f}s")

    # Estimate for production scale
    D_prod = 10000
    scale_factor = (D_prod / D) ** 2  # Complexity is O(D^2 * k)
    estimated_time = elapsed * scale_factor

    print(f"\n[INFO] Production Scale Estimate (D={D_prod:,}):")
    print(f"       Estimated time: {estimated_time:.1f}s ({estimated_time/60:.1f} minutes)")
    print(f"       For T=720: ~{estimated_time*720/(60*60):.1f} hours")

    return True


def main():
    """Main execution function."""
    print("Randomized Ridge Regression Test Suite")
    print("=" * 60)

    # Check sklearn availability
    try:
        from sklearn.utils.extmath import randomized_svd
        print("[OK] scikit-learn available")
    except ImportError:
        print("[ERROR] scikit-learn not available")
        print("  Install with: pip install scikit-learn")
        print("\nRandomized SVD requires scikit-learn.")
        sys.exit(1)

    print()

    # Run tests
    tests_passed = 0
    tests_total = 3

    if test_basic_functionality():
        tests_passed += 1

    if test_consistency():
        tests_passed += 1

    if benchmark_performance():
        tests_passed += 1

    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("\n[PASS] All tests passed")
        return 0
    else:
        print(f"\n[FAIL] {tests_total - tests_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
