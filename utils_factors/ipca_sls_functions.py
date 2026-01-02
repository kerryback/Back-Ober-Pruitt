"""
IPCA (Instrumented Principal Component Analysis) estimation using Sequential Least Squares (SLS).

This implementation uses alternating least squares optimization instead of pymanopt/Riemannian optimization.
While slower than the Stiefel manifold approach, it requires no external optimization libraries.

Key Features:
- Alternating least squares: iteratively solve for factors then loadings
- QR orthonormalization to maintain Gamma'Gamma = I_K constraint
- Sign normalization (factors have positive mean)
- Warm-starting for rolling windows

References:
    Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics are covariances:
    A unified model of risk and return. Journal of Financial Economics, 134(3), 501-524.
"""

import numpy as np
from scipy.linalg import solve, qr
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

# Import rank standardization from factor_utils
from .factor_utils import rank_standardize


def precompute_sufficient_stats(
    panel: pd.DataFrame,
    start: int,
    end: int,
    chars: List[str]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Precompute sufficient statistics for IPCA estimation.

    This reduces complexity from O(T·N·L) to O(T·L²) per optimization iteration,
    which is critical for large panels.

    Args:
        panel: Panel data with multi-index (month, firmid)
        start: Start month (inclusive)
        end: End month (inclusive)
        chars: List of characteristic column names

    Returns:
        Ws: (T, L, L) array where Ws[t] = Z_t' @ Z_t
        xs: (T, L) array where xs[t] = Z_t' @ r_t
        total_ss: Total sum of squares of returns (for R² calculation)

    Note:
        Z_t is the (N_t, L) matrix of characteristics at month t
        r_t is the (N_t,) vector of excess returns at month t
    """
    T = end - start + 1
    L = len(chars)

    Ws = np.zeros((T, L, L))
    xs = np.zeros((T, L))
    total_ss = 0.0

    for i, t in enumerate(range(start, end + 1)):
        data = panel.loc[t]
        Z_t = data[chars].to_numpy()  # (N_t, L)
        r_t = data['xret'].to_numpy()  # (N_t,)

        # Sufficient statistics (much smaller than original data!)
        Ws[i] = Z_t.T @ Z_t  # (L, L) - small!
        xs[i] = Z_t.T @ r_t  # (L,) - tiny!
        total_ss += np.dot(r_t, r_t)

    return Ws, xs, total_ss


def fit_ipca_sls_single(
    Ws: np.ndarray,
    xs: np.ndarray,
    K: int,
    max_iterations: int = 1000,
    tol: float = 1e-6,
    Gamma_init: Optional[np.ndarray] = None,
    verbosity: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit IPCA using Sequential Least Squares (alternating optimization).

    Alternates between:
    1. Fix Gamma, solve for factors: f_t = (Gamma' W_t Gamma)^{-1} Gamma' x_t
    2. Fix factors, solve for Gamma: Gamma = (sum_t W_t)^{-1} (sum_t x_t f_t')
       then orthonormalize via QR decomposition

    Args:
        Ws: (T, L, L) sufficient statistics Z_t' @ Z_t
        xs: (T, L) sufficient statistics Z_t' @ r_t
        K: Number of latent factors
        max_iterations: Maximum alternating optimization iterations
        tol: Convergence tolerance (relative change in objective)
        Gamma_init: Initial Gamma (if None, uses SVD-based initialization)
        verbosity: 0=silent, 1=minimal, 2=detailed

    Returns:
        Gamma: (L, K) loading coefficient matrix (Gamma'Gamma = I_K)
        factors: (K, T) matrix of factor realizations
        pi: (K,) portfolio weights for mean-variance efficient factor portfolio
        info: Dictionary with convergence information
    """
    T, L, _ = Ws.shape

    # Initialize Gamma using data-driven SVD if not provided
    # This matches the original IPCA code initialization (Kelly et al. 2019, pp. 507)
    if Gamma_init is None:
        # Compute cross-sectional averages: X[t,:] = Z_t' r_t / N_t
        # N_t is the last diagonal element of Ws[t] (from the 'ones' column)
        X = np.zeros((T, L))
        for t in range(T):
            N_t = Ws[t][-1, -1]  # Intercept column diagonal = N_t
            X[t, :] = xs[t] / N_t

        # SVD of X.T to get principal components
        U, _, _ = np.linalg.svd(X.T, full_matrices=False)
        Gamma = U[:, :K]  # First K left singular vectors

        if verbosity >= 2:
            print(f"    SVD initialization: Gamma shape = {Gamma.shape}")
    else:
        Gamma = Gamma_init.copy()

    # Track objective value (RSS)
    prev_obj = np.inf

    for iteration in range(max_iterations):
        # Step 1: Fix Gamma, solve for factors
        # f_t = (Gamma' W_t Gamma)^{-1} Gamma' x_t
        factors = np.zeros((K, T))
        for t in range(T):
            GtWG = Gamma.T @ Ws[t] @ Gamma  # (K, K)
            Gtx = Gamma.T @ xs[t]  # (K,)
            factors[:, t] = solve(GtWG, Gtx, assume_a='pos')

        # Step 2: Fix factors, solve for Gamma
        # Minimize: sum_t ||x_t - W_t Gamma f_t||^2
        # Solution: Gamma = (sum_t W_t)^{-1} (sum_t x_t f_t')
        # But this doesn't enforce orthonormality, so we use normal equations:

        # Build normal equations for vec(Gamma)
        # We want: Gamma = argmin sum_t ||x_t - W_t Gamma f_t||^2
        # Normal equations: (sum_t W_t (f_t f_t') W_t') vec(Gamma) = sum_t W_t (x_t f_t')
        # Simplification: solve for each column of Gamma separately

        # Actually, better approach: solve sum_t W_t Gamma f_t f_t' = sum_t x_t f_t'
        # Gamma = (sum_t W_t)^{-1} (sum_t x_t f_t') (sum_t f_t f_t')^{-1}

        sum_W = np.sum(Ws, axis=0)  # (L, L)
        sum_xf = np.sum([np.outer(xs[t], factors[:, t]) for t in range(T)], axis=0)  # (L, K)
        sum_ff = factors @ factors.T  # (K, K)

        # Solve: sum_W @ Gamma = sum_xf @ inv(sum_ff')
        Gamma_new = solve(sum_W, sum_xf @ np.linalg.inv(sum_ff), assume_a='pos')

        # Orthonormalize Gamma via QR decomposition
        Gamma, _ = qr(Gamma_new, mode='economic')  # economic mode gives (L, K) not (L, L)

        # Recompute factors with new Gamma
        for t in range(T):
            GtWG = Gamma.T @ Ws[t] @ Gamma
            Gtx = Gamma.T @ xs[t]
            factors[:, t] = solve(GtWG, Gtx, assume_a='pos')

        # Compute objective (RSS)
        obj = 0.0
        for t in range(T):
            W_t = Ws[t]
            x_t = xs[t]
            AtWA = Gamma.T @ (W_t @ Gamma)
            Atx = Gamma.T @ x_t
            gamma_t = solve(AtWA, Atx, assume_a='pos')
            obj += np.dot(x_t, x_t) - np.dot(Atx, gamma_t)

        # Check convergence
        rel_change = abs(obj - prev_obj) / (abs(prev_obj) + 1e-10)

        if verbosity >= 2:
            print(f"    Iter {iteration + 1}: obj = {obj:.6f}, rel_change = {rel_change:.2e}")

        if rel_change < tol:
            if verbosity >= 1:
                print(f"    Converged in {iteration + 1} iterations")
            break

        prev_obj = obj
    else:
        if verbosity >= 1:
            print(f"    Warning: Max iterations ({max_iterations}) reached without convergence")

    # Final normalization (Kelly et al. 2019, pp. 507)
    # Ensures: (1) Gamma' @ Gamma = I, (2) f @ f.T is diagonal, (3) positive mean
    if K < L:  # Only normalize if K < L (for K == L, Gamma is identity)
        # Step 1: Orthonormalize Gamma using Cholesky
        chol = np.linalg.cholesky(Gamma.T @ Gamma)
        cholinv = np.linalg.inv(chol)

        # Step 2: Diagonalize factor covariance
        U, _, _ = np.linalg.svd(chol @ factors @ factors.T @ chol.T)

        # Step 3: Apply rotation
        Gamma = Gamma @ cholinv @ U
        factors = U.T @ chol @ factors

    # Sign normalization (convention: each factor should have positive mean)
    sign_conv = np.sign(np.mean(factors, axis=1)).reshape((-1, 1))
    sign_conv[sign_conv == 0] = 1  # Handle zero means
    factors = factors * sign_conv
    Gamma = Gamma * sign_conv.T

    # Compute mean-variance efficient portfolio of factors
    # Solves: min ||f'π - 1||² → π = (f f')^{-1} f 1
    pi = np.linalg.lstsq(factors.T, np.ones(T), rcond=None)[0]

    # Convergence info
    info = {
        'cost': obj,
        'iterations': iteration + 1,
        'converged': rel_change < tol
    }

    return Gamma, factors, pi, info


def fit_ipca_sls(
    panel: pd.DataFrame,
    start: int,
    K: int,
    chars: List[str],
    n_restarts: int = 3,
    max_iterations: int = 1000,
    verbosity: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit IPCA using Sequential Least Squares with multiple random restarts.

    Args:
        panel: Panel data with multi-index (month, firmid)
        start: Start month for 360-month estimation window
        K: Number of latent factors
        chars: List of characteristic column names
        n_restarts: Number of random restarts (higher = more robust)
        max_iterations: Maximum iterations per restart
        verbosity: 0=silent, 1=minimal, 2=detailed

    Returns:
        Gamma: (L, K) loading coefficient matrix (Γ'Γ = I_K)
        factors: (K, T) matrix of factor realizations
        pi: (K,) portfolio weights for mean-variance efficient factor portfolio
        info: Dictionary with convergence information
    """
    L = len(chars)
    if K > L:
        raise ValueError(f"K={K} cannot exceed L={L} (number of characteristics)")

    # Precompute sufficient statistics for 360-month window
    Ws, xs, total_ss = precompute_sufficient_stats(
        panel, start, start + 359, chars
    )

    # Multiple random restarts
    best_Gamma, best_factors, best_pi, best_info = None, None, None, None
    best_cost = np.inf

    for restart in range(n_restarts):
        if verbosity >= 1:
            print(f"  Restart {restart + 1}/{n_restarts}")

        Gamma, factors, pi, info = fit_ipca_sls_single(
            Ws, xs, K,
            max_iterations=max_iterations,
            verbosity=max(0, verbosity - 1)
        )

        if info['cost'] < best_cost:
            best_cost = info['cost']
            best_Gamma = Gamma
            best_factors = factors
            best_pi = pi
            best_info = info

            if verbosity >= 1:
                print(f"    New best: cost = {info['cost']:.2f}, iterations = {info['iterations']}")

    best_info['n_restarts'] = n_restarts
    return best_Gamma, best_factors, best_pi, best_info


def fit_ipca_rolling(
    panel: pd.DataFrame,
    K: int,
    N: int,
    start: int,
    end: int,
    chars: List[str],
    n_restarts: int = 3,
    verbosity: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Fit IPCA using rolling 360-month windows.

    This is the main IPCA estimation function, computing loadings and
    factor portfolios for each month using the previous 360 months of data.

    Args:
        panel: Panel with multi-index (month, firmid)
        K: Number of factors
        N: Number of firms (for output array sizing)
        start: First month for factor computation (needs 360 months before)
        end: Last month for factor computation
        chars: Characteristic names (will be rank-standardized)
        n_restarts: Random restarts for first window (subsequent use 1 for warm start)
        verbosity: 0=silent, 1=minimal, 2=detailed

    Returns:
        ipca_weights: (K, N, n_windows) array of factor loadings Z_t @ Γ_t
        ipca_pi: (K, n_windows) array of factor portfolio weights
        info_list: List of convergence info dicts for each window

    Note:
        - Characteristics are automatically rank-standardized
        - An intercept ('ones') is added to characteristics
        - First window uses n_restarts, subsequent windows use 1 (warm start)
    """
    if verbosity > 0:
        print(f"Fitting rolling IPCA (SLS): K={K}, {start} to {end}")

    # Rank-standardize characteristics and add intercept
    panel_ranked = panel[chars].groupby('month').apply(
        lambda g: pd.DataFrame(
            rank_standardize(g.to_numpy()),
            index=g.index,
            columns=g.columns
        )
    ).reset_index(level=0, drop=True)

    panel_ranked['ones'] = 1.0
    panel_ranked['xret'] = panel['xret']

    # Use rank-standardized panel for IPCA
    panel = panel_ranked
    chars_with_ones = chars + ['ones']

    n_windows = end + 1 - start - 360
    ipca_weights = np.zeros((K, N, n_windows))
    ipca_pi = np.zeros((K, n_windows))
    info_list = []

    # First window with multiple restarts
    if verbosity > 0:
        print(f"  Window 1/{n_windows}: months {start} to {start + 359}")

    Gamma0, f0, pi0, info0 = fit_ipca_sls(
        panel, start, K, chars_with_ones,
        n_restarts=n_restarts,
        verbosity=max(0, verbosity - 1)
    )

    # Compute factor loadings for month start+360
    data = panel.loc[start + 360]
    firms = data.index.to_numpy()
    Z_360 = data[chars_with_ones].to_numpy()

    # Factor loadings: X = Z @ Γ, then factor_port = (X.T @ X)^{-1} @ X.T
    # This matches the original IPCA code (Kelly et al. 2019)
    X = Z_360 @ Gamma0  # (N_t, K)
    factor_port = np.linalg.pinv(X.T @ X) @ X.T  # (K, N_t) - scaled portfolio weights
    ipca_weights[:, firms, 0] = factor_port
    ipca_pi[:, 0] = pi0
    info_list.append(info0)

    # Save previous Gamma for warm start
    prev_Gamma = Gamma0

    # Rolling windows (use warm start = 1 restart with previous Gamma)
    for i, t in enumerate(range(start + 1, end + 1 - 360), start=1):
        if verbosity > 0:
            if i % 50 == 0 or i == n_windows - 1:
                print(f"  Window {i + 1}/{n_windows}: months {t} to {t + 359}")

        # Warm start with previous Gamma
        Ws, xs, _ = precompute_sufficient_stats(panel, t, t + 359, chars_with_ones)
        Gamma_t, f_t, pi_t, info_t = fit_ipca_sls_single(
            Ws, xs, K,
            Gamma_init=prev_Gamma,  # Warm start
            verbosity=0
        )

        # Compute factor loadings for month t+360
        data = panel.loc[t + 360]
        firms = data.index.to_numpy()
        Z_t = data[chars_with_ones].to_numpy()

        # Factor loadings: X = Z @ Γ, then factor_port = (X.T @ X)^{-1} @ X.T
        X = Z_t @ Gamma_t  # (N_t, K)
        factor_port = np.linalg.pinv(X.T @ X) @ X.T  # (K, N_t)
        ipca_weights[:, firms, i] = factor_port
        ipca_pi[:, i] = pi_t
        info_list.append(info_t)

        # Update previous Gamma for next warm start
        prev_Gamma = Gamma_t

    if verbosity > 0:
        avg_iters = np.mean([info['iterations'] for info in info_list])
        print(f"  Average iterations per window: {avg_iters:.1f}")

    return ipca_weights, ipca_pi, info_list


# Export main functions
__all__ = [
    'precompute_sufficient_stats',
    'fit_ipca_sls_single',
    'fit_ipca_sls',
    'fit_ipca_rolling',
]
