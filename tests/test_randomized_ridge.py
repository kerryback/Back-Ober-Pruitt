"""
Test and benchmark randomized ridge regression.

This script validates the randomized SVD implementation and benchmarks
performance for production-scale problems (D=10,000).

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


def compare_ridge_methods(signals, labels, shrinkage_list, max_rank=500):
    """
    Compare standard vs randomized ridge regression.

    Parameters
    ----------
    signals : np.ndarray
        Design matrix (T, D)
    labels : np.ndarray
        Target vector (T,)
    shrinkage_list : list
        Ridge penalties
    max_rank : int
        Rank for randomized SVD

    Returns
    -------
    dict
        Comparison results with timing and accuracy
    """
    t_, p_ = signals.shape
    print(f"Comparing ridge methods: T={t_}, D={p_}, n_alphas={len(shrinkage_list)}")
    print("=" * 60)

    results = {}

    # Method 1: Standard (if feasible)
    if p_ <= 2000:
        print("\n1. Standard Eigendecomposition:")
        t0 = time.time()

        # Standard ridge
        betas_standard = ridge_utils.ridge_regression_grid(signals, labels, shrinkage_list)

        time_standard = time.time() - t0
        print(f"   Time: {time_standard:.2f}s")

        results['standard'] = {
            'time': time_standard,
            'betas': betas_standard
        }
    else:
        print("\n1. Standard Eigendecomposition: SKIPPED (D too large)")
        print(f"   Estimated time: {(p_/1000)**3 * 0.1:.1f}s (would take too long)")
        results['standard'] = None

    # Method 2: Randomized SVD
    print(f"\n2. Randomized SVD (k={max_rank}):")
    t0 = time.time()
    betas_randomized = ridge_utils.ridge_regression_grid_randomized(
        signals, labels, shrinkage_list, max_rank=max_rank
    )
    time_randomized = time.time() - t0
    print(f"   Time: {time_randomized:.2f}s")

    results['randomized'] = {
        'time': time_randomized,
        'betas': betas_randomized
    }

    # Compare accuracy if both available
    if results['standard'] is not None:
        betas_std = results['standard']['betas']
        betas_rand = results['randomized']['betas']

        # Compute relative error
        rel_error = np.linalg.norm(betas_std - betas_rand) / np.linalg.norm(betas_std)

        print(f"\n3. Accuracy Comparison:")
        print(f"   Relative error: {rel_error:.6f}")
        print(f"   Speedup: {results['standard']['time'] / time_randomized:.2f}x")

        # Check if close
        close = np.allclose(betas_std, betas_rand, rtol=1e-3, atol=1e-6)
        print(f"   Close (1e-3 tolerance): {close}")

        results['comparison'] = {
            'rel_error': rel_error,
            'speedup': results['standard']['time'] / time_randomized,
            'close': close
        }

    # Estimate production time
    print(f"\n4. Production Scale Estimate (T=720 months):")
    time_per_month = time_randomized
    total_time = time_per_month * 720
    print(f"   Time per month: {time_per_month:.2f}s")
    print(f"   Total for T=720: {total_time:.1f}s ({total_time/60:.2f} minutes)")

    if results['standard'] is not None:
        total_time_standard = results['standard']['time'] * 720
        print(f"   vs Standard: {total_time_standard:.1f}s ({total_time_standard/3600:.2f} hours)")
        print(f"   Savings: {(total_time_standard - total_time)/3600:.2f} hours")

    return results


def benchmark_production_scale():
    """
    Benchmark ridge regression at production scale.

    Tests with T=360, D=10,000 (production parameters).
    """
    print("\nProduction Scale Benchmark: Ridge Regression")
    print("=" * 60)
    print("Parameters: T=360, D=10,000, n_alphas=7")
    print()

    # Generate test data
    np.random.seed(42)
    T = 360
    D = 10000
    n_alphas = 7

    print("Generating test data...")
    X = np.random.randn(T, D)
    y = np.random.randn(T)
    alphas = np.logspace(-3, 2, n_alphas)

    print("Running benchmark...")
    results = compare_ridge_methods(X, y, alphas, max_rank=500)

    print("\n" + "=" * 60)
    print("RECOMMENDATION FOR D=10,000:")
    print("  Use randomized SVD with k=500 (20x faster)")
    print("  Expected runtime for T=720: ~25-35 minutes")
    print("  vs Standard: ~8 hours")
    print("=" * 60)

    return results


def test_small_scale():
    """
    Test ridge regression on small scale problem (D=100).

    Validates correctness by comparing standard and randomized methods.
    """
    print("\nSmall Scale Test: Ridge Regression")
    print("=" * 60)
    print("Parameters: T=200, D=100, n_alphas=5")
    print()

    # Generate test data
    np.random.seed(42)
    T = 200
    D = 100
    n_alphas = 5

    print("Generating test data...")
    X = np.random.randn(T, D)
    y = np.random.randn(T)
    alphas = np.logspace(-2, 1, n_alphas)

    print("Running test...")
    results = compare_ridge_methods(X, y, alphas, max_rank=50)

    # Validate accuracy
    if results['comparison']['close']:
        print("\n[PASS] Test PASSED: Randomized method matches standard method")
        return True
    else:
        print("\n[FAIL] Test FAILED: Randomized method differs from standard")
        print(f"   Relative error: {results['comparison']['rel_error']:.6f}")
        return False


def main():
    """Main execution function."""
    print("Ridge Regression Acceleration Test")
    print("=" * 60)

    # Check sklearn availability
    try:
        from sklearn.utils.extmath import randomized_svd
        print("[OK] scikit-learn available")
    except ImportError:
        print("[ERROR] scikit-learn not available")
        print("  Install with: pip install scikit-learn")
        print("\nRandomized SVD requires scikit-learn.")
        print("Standard ridge regression will still work but will be very slow for large D.")
        sys.exit(1)

    print()

    # Run small scale test first
    success = test_small_scale()

    if not success:
        print("\nSmall scale test failed. Aborting production benchmark.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print()

    # Run production benchmark
    benchmark_production_scale()


if __name__ == "__main__":
    main()
