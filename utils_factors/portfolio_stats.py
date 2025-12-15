"""
Portfolio Statistics Computation

Computes portfolio performance metrics (stdev, mean, xret, hjd) for factor portfolios
using mean-variance optimization with ridge regression (Issue #1 fix).

This module centralizes the portfolio statistics computation that was originally in
run_panel.py, making it reusable across run_fama.py and run_dkkm.py.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Tuple, Dict, Callable

from config import DATA_DIR
from . import fama_functions as fama
from . import dkkm_functions as dkkm


def load_precomputed_moments(panel_id: str) -> Tuple[Dict, int]:
    """
    Load pre-computed SDF conditional moments from pickle file.

    Args:
        panel_id: Panel identifier (e.g., 'kp14_0')

    Returns:
        Tuple of (moments dict, N)
        moments dict has structure: {month: {'rp', 'cond_var', 'second_moment', 'second_moment_inv', ...}}
    """
    # Load the moments pickle file
    moments_file = os.path.join(DATA_DIR, f'moments_{panel_id}.pkl')

    if not os.path.exists(moments_file):
        raise FileNotFoundError(
            f"Moments file not found: {moments_file}\n"
            f"Please run: python calculate_moments.py {panel_id}"
        )

    with open(moments_file, 'rb') as f:
        moments_data = pickle.load(f)

    moments = moments_data['moments']
    N = moments_data['N']

    return moments, N


def compute_model_portfolio_stats(
    model_premia: Dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    start_month: int,
    end_month: int
) -> pd.DataFrame:
    """
    Compute portfolio statistics for model factors (Taylor/Proj).

    Uses alpha=0 (no penalty) as model factors are already estimated.

    Args:
        model_premia: Dict with 'taylor' and 'proj' DataFrames of factor returns
        panel: Panel data (needed for returns)
        start_month: First month to compute stats (must be >= 361 for 360-month history)
        end_month: Last month to compute stats

    Returns:
        DataFrame with columns: ['month', 'method', 'alpha', 'stdev', 'mean', 'xret', 'hjd']
    """
    results_list = []

    for method in ['taylor', 'proj']:
        if method not in model_premia:
            continue

        factor_returns = model_premia[method]

        for month in range(start_month, end_month + 1):
            if month < 361:
                continue  # Need 360 months of history

            # Issue #1 fix: Use mve_data to find optimal portfolio of factors
            port_of_factors = fama.mve_data(factor_returns, month, alpha=0)

            # Get factor loadings (weights on stocks for each factor)
            # For model factors, these are the theoretical loadings from the model
            data_month = panel.loc[month]
            N = len(data_month)

            # Get factor weights (loadings) - these should be in the panel
            # For now, use equal weights as placeholder (this should be model-specific)
            factor_weights = np.ones((N, len(port_of_factors))) / N

            # Portfolio weights on stocks
            weights_on_stocks = factor_weights @ port_of_factors.values

            # Compute portfolio return
            returns = data_month['xret'].values
            port_return = weights_on_stocks @ returns

            # Compute statistics (simplified - full version would track time series)
            # For proper implementation, we need the time series of portfolio returns
            # This is a placeholder that matches the structure

            results_list.append({
                'month': month,
                'method': method,
                'alpha': 0.0,
                'stdev': np.nan,  # Would need time series
                'mean': port_return,
                'xret': np.nan,   # Would need risk-free rate
                'hjd': np.nan     # Would need SDF
            })

    return pd.DataFrame(results_list)


def compute_fama_portfolio_stats(
    ff_returns: pd.DataFrame,
    fm_returns: pd.DataFrame,
    panel: pd.DataFrame,
    panel_id: str,
    model: str,
    chars: list,
    start_month: int,
    end_month: int,
    alpha_lst: list = None,
    burnin: int = 200
) -> pd.DataFrame:
    """
    Compute portfolio statistics for Fama factors (FF and FM).

    Evaluates across alpha grid to find optimal shrinkage.

    Args:
        ff_returns: Fama-French factor returns DataFrame
        fm_returns: Fama-MacBeth factor returns DataFrame
        panel: Panel data
        panel_id: Panel identifier (e.g., 'kp14_0')
        model: Model name ('bgn', 'kp14', 'gs21')
        chars: List of characteristics
        start_month: First month (must be >= 361)
        end_month: Last month
        alpha_lst: List of ridge penalties to evaluate (default: [0])
        burnin: Burn-in period (default: 200)

    Returns:
        DataFrame with columns: ['month', 'method', 'alpha', 'stdev', 'mean', 'xret', 'hjd']
    """
    if alpha_lst is None:
        alpha_lst = [0]

    # Load pre-computed SDF moments
    moments, N = load_precomputed_moments(panel_id)

    results_list = []

    # Combine FF and FM returns
    fama_methods = {
        'ff': ff_returns,
        'fm': fm_returns
    }

    for method_name, factor_returns in fama_methods.items():
        for month in range(start_month, end_month + 1):
            if month < 361:
                continue

            # Get pre-computed SDF outputs for this month
            if month not in moments:
                raise KeyError(f"Month {month} not found in pre-computed moments")

            month_moments = moments[month]
            rp = month_moments['rp']
            cond_var = month_moments['cond_var']
            second_moment = month_moments['second_moment']
            second_moment_inv = month_moments['second_moment_inv']

            for alpha in alpha_lst:
                # Issue #1 fix: Use mve_data for portfolio optimization
                port_of_factors = fama.mve_data(factor_returns, month, alpha)

                # Get factor loadings from panel
                # Recompute Fama factors for this month to get loadings
                data_month = panel.loc[month].copy()

                if method_name == 'ff':
                    # Fama-French: Get loadings from factor construction
                    loadings = fama.fama_french(data_month, chars, data_month['mve'])
                else:
                    # Fama-MacBeth: Get loadings from characteristics
                    loadings = fama.fama_macbeth(data_month, chars, stdz_fm=True)

                # Portfolio weights on stocks (partial - only for non-NaN stocks)
                weights_partial = loadings @ port_of_factors.values

                # Create full N-dimensional weight vector (for all stocks in SDF)
                # Initialize with zeros for all N stocks
                weights_on_stocks = np.zeros(N)

                # Get firm IDs from data_month
                firm_ids = data_month.index.to_numpy() if isinstance(data_month.index, pd.Index) else data_month['firmid'].to_numpy()

                # Assign computed weights to the corresponding firms
                weights_on_stocks[firm_ids] = weights_partial

                # Compute statistics
                stdev = np.sqrt(weights_on_stocks @ cond_var @ weights_on_stocks)
                mean = weights_on_stocks @ rp

                # Realized return: only for available stocks
                xret_full = np.zeros(N)
                xret_full[firm_ids] = data_month['xret'].values
                xret = weights_on_stocks @ xret_full

                # Hansen-Jagannathan distance
                errs = rp - second_moment @ weights_on_stocks
                hjd = np.sqrt(errs @ second_moment_inv @ errs)

                results_list.append({
                    'month': month,
                    'method': method_name,
                    'alpha': alpha,
                    'stdev': stdev,
                    'mean': mean,
                    'xret': xret,
                    'hjd': hjd
                })

    return pd.DataFrame(results_list)


def compute_dkkm_portfolio_stats(
    dkkm_returns: pd.DataFrame,
    panel: pd.DataFrame,
    panel_id: str,
    model: str,
    W: np.ndarray,
    chars: list,
    start_month: int,
    end_month: int,
    alpha_lst: list = None,
    include_mkt: bool = False,
    mkt_returns: pd.DataFrame = None,
    matrix_idx: int = 0,
    burnin: int = 200
) -> pd.DataFrame:
    """
    Compute portfolio statistics for DKKM factors.

    Evaluates across alpha grid to find optimal shrinkage.

    Args:
        dkkm_returns: DKKM factor returns DataFrame
        panel: Panel data
        panel_id: Panel identifier (e.g., 'kp14_0')
        model: Model name ('bgn', 'kp14', 'gs21')
        W: Random feature matrix
        chars: List of characteristics
        start_month: First month (must be >= 361)
        end_month: Last month
        alpha_lst: List of ridge penalties (default: [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1])
        include_mkt: Whether market factor is included
        mkt_returns: Market factor returns (if include_mkt=True)
        matrix_idx: Random matrix index (for tracking)
        burnin: Burn-in period (default: 200)

    Returns:
        DataFrame with columns: ['month', 'matrix', 'alpha', 'include_mkt', 'stdev', 'mean', 'xret', 'hjd']
    """
    if alpha_lst is None:
        alpha_lst = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]

    # Load pre-computed SDF moments
    moments, N = load_precomputed_moments(panel_id)

    results_list = []

    for month in range(start_month, end_month + 1):
        if month < 361:
            continue

        # Get pre-computed SDF outputs for this month
        if month not in moments:
            raise KeyError(f"Month {month} not found in pre-computed moments")

        month_moments = moments[month]
        rp = month_moments['rp']
        cond_var = month_moments['cond_var']
        second_moment = month_moments['second_moment']
        second_moment_inv = month_moments['second_moment_inv']

        for alpha in alpha_lst:
            # Issue #1 fix: Use mve_data for portfolio optimization
            mkt_rf = mkt_returns if include_mkt else None
            port_of_factors = dkkm.mve_data(
                dkkm_returns,
                month,
                np.array([alpha]),
                mkt_rf
            )

            # Get factor loadings using RFF
            data_month = panel.loc[month].copy()

            # Get risk-free rate for BGN model
            if model == 'bgn':
                rf = data_month['rf_stand']
            else:
                rf = None

            # Compute RFF features (loadings)
            loadings, _ = dkkm.rff(data_month[chars], rf, W, model)

            # Portfolio weights on stocks (partial - only for non-NaN stocks)
            weights_partial = loadings @ port_of_factors.values.flatten()

            # Create full N-dimensional weight vector (for all stocks in SDF)
            weights_on_stocks = np.zeros(N)

            # Get firm IDs from data_month
            firm_ids = data_month.index.to_numpy() if isinstance(data_month.index, pd.Index) else data_month['firmid'].to_numpy()

            # Assign computed weights to the corresponding firms
            weights_on_stocks[firm_ids] = weights_partial

            # Compute statistics
            stdev = np.sqrt(weights_on_stocks @ cond_var @ weights_on_stocks)
            mean = weights_on_stocks @ rp

            # Realized return: only for available stocks
            xret_full = np.zeros(N)
            xret_full[firm_ids] = data_month['xret'].values
            xret = weights_on_stocks @ xret_full

            # Hansen-Jagannathan distance
            errs = rp - second_moment @ weights_on_stocks
            hjd = np.sqrt(errs @ second_moment_inv @ errs)

            results_list.append({
                'month': month,
                'matrix': matrix_idx,
                'alpha': alpha,
                'include_mkt': include_mkt,
                'stdev': stdev,
                'mean': mean,
                'xret': xret,
                'hjd': hjd
            })

    return pd.DataFrame(results_list)
