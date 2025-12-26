# run_ipca.py Pickle File Contents

**File location**: `outputs/{panel_id}_ipca_{K}.pkl`
**Created by**: `run_ipca.py`
**Created at**: [run_ipca.py:268-280](run_ipca.py#L268-L280)

## Overview

The IPCA (Instrumented Principal Component Analysis) pickle file contains latent factor returns estimated using Stiefel manifold optimization, along with factor loadings (weights on stocks), portfolio weights, convergence information, and performance statistics.

## Items Included (10 total)

---

### 1. `'ipca_factors'` - IPCA Factor Returns

- **Type**: `pd.DataFrame`
- **Shape**: `(T-360, K)` where T = total months, K = number of latent factors
- **Columns**: `['ipca_1', 'ipca_2', ..., 'ipca_K']`
- **Index**: Month numbers starting from `start + 360`

**Creation trace**:
1. [run_ipca.py:102-148](run_ipca.py#L102-L148): `compute_ipca_factor_returns()` function
   - Loops through months from `start + 360` to `end`
   - For each month:
     - Gets stock returns: `r_t = data['xret']`
     - Gets factor loadings: `loadings_t = ipca_weights[:, firms, i]` (K × N_t)
     - Computes factor returns: `f_t = loadings_t @ r_t` (K,)
   - Creates DataFrame with column names `['ipca_1', 'ipca_2', ...]`
2. **Loadings** come from `ipca_weights` computed by:
   - [run_ipca.py:79-88](run_ipca.py#L79-L88): Calls `ipca.fit_ipca_rolling()`
   - [ipca_functions.py:273-370](utils_factors/ipca_functions.py#L273-L370): `fit_ipca_rolling()` function
     - Uses 360-month rolling windows
     - For each window, calls `fit_ipca_single_window()`
     - Optimizes on Stiefel manifold using pymanopt
     - Returns loadings for month t+360

**Contents**: Monthly returns for IPCA latent factors

**Note**:
- Factor returns start at month `start + 360` because IPCA uses 360-month estimation window
- Number of observations is `T - 360` where T is the total number of months
- Factors are orthogonalized: Gamma'Gamma = I

---

### 2. `'ipca_weights'` - Factor Loadings (Weights on Stocks)

- **Type**: `np.ndarray`
- **Shape**: `(K, N, n_windows)` where:
  - K = number of latent factors
  - N = total number of firms (max across all months)
  - n_windows = number of rolling windows = T - 360
- **Values**: Factor loadings β_{it,k} for firm i, factor k, window t

**Creation trace**:
1. [run_ipca.py:79-88](run_ipca.py#L79-L88): Called via `ipca.fit_ipca_rolling()`
2. [ipca_functions.py:273-370](utils_factors/ipca_functions.py#L273-L370): `fit_ipca_rolling()` function
   - Pre-allocates array: `shape (K, N, n_windows)`
   - For each rolling window (from start to end-360):
     - Calls `fit_ipca_single_window()` to estimate Gamma and f for 360 months
     - Computes loadings for month t+360: `loadings = X @ Gamma` where X are characteristics
     - Stores in `ipca_weights[:, firm_ids, window_idx]`
3. [ipca_functions.py:94-270](utils_factors/ipca_functions.py#L94-L270): `fit_ipca_single_window()` function
   - Optimizes on Stiefel manifold (Gamma such that Gamma'Gamma = I)
   - Uses pymanopt with conjugate gradient
   - Minimizes: ||R - (X ⊗ f') Gamma||² over 360-month window
   - Returns Gamma (L × K) and f (K × 360)

**Contents**: Factor loadings for each firm at each month

**Structure**:
- Dimension 0: Factor index (0 to K-1)
- Dimension 1: Firm index (0 to N-1)
- Dimension 2: Window/time index (0 to n_windows-1)

**Note**: Many entries are zero because not all firms exist at all times

---

### 3. `'ipca_pi'` - Portfolio Weights on Factors

- **Type**: `np.ndarray`
- **Shape**: `(K, n_windows)` where K = latent factors, n_windows = T - 360
- **Values**: Portfolio weights π_k for combining factors

**Creation trace**:
1. [run_ipca.py:79-88](run_ipca.py#L79-L88): Returned from `ipca.fit_ipca_rolling()`
2. [ipca_functions.py:339-344](utils_factors/ipca_functions.py#L339-L344): Computed in `fit_ipca_rolling()`
   - Computes ridge regression: minimize ||1 - f'π||² with no penalty (alpha=0)
   - `pi = ridge_regression_fast(f.T, ones, alpha=0)` where f is (K, 360)
   - Finds portfolio of factors that best replicates constant return

**Contents**: Weights for mean-variance efficient portfolio of IPCA factors

**Usage**: Can be used to construct a single "IPCA portfolio" from the K factors:
```python
ipca_portfolio = ipca_factors @ ipca_pi[:, t]  # For window t
```

---

### 4. `'ipca_stats'` - Portfolio Statistics for IPCA Factors

- **Type**: `pd.DataFrame`
- **Shape**: `(n_months * n_alphas, 8)` where n_months = months with sufficient history
- **Columns**: `['month', 'K', 'alpha', 'include_mkt', 'stdev', 'mean', 'xret', 'hjd']`
- **Index**: Default range index

**Creation trace**:
1. [run_ipca.py:242-256](run_ipca.py#L242-L256): Called via `portfolio_stats.compute_ipca_portfolio_stats()`
2. [portfolio_stats.py:357-521](utils_factors/portfolio_stats.py#L357-L521): `compute_ipca_portfolio_stats()` function
   - Loads pre-computed SDF moments from `{panel_id}_moments.pkl`
   - Loops over months with sufficient history (at least 100 past months)
   - Loops over alpha values (ridge penalties)
   - For each combination:
     - [portfolio_stats.py:424-467](utils_factors/portfolio_stats.py#L424-L467): Computes optimal portfolio using ridge regression
       - Uses past 360 months of IPCA factor returns
       - Optional market factor inclusion
       - Returns portfolio weights on factors
     - [portfolio_stats.py:469-519](utils_factors/portfolio_stats.py#L469-L519): Computes portfolio statistics
       - Extracts IPCA loadings from ipca_weights array for current window
       - Computes portfolio weights on stocks: `weights_on_stocks = loadings @ port_of_factors`
       - Computes full statistics using SDF moments

**Column descriptions**:
- `month`: Month number
- `K`: Number of latent factors
- `alpha`: Ridge penalty parameter
- `include_mkt`: Whether market factor is included
- `stdev`: Portfolio standard deviation (from conditional variance: √(w'Σw))
- `mean`: Portfolio expected return (from risk premium: w'μ)
- `xret`: Portfolio realized excess return (from actual stock returns)
- `hjd`: Hansen-Jagannathan distance (pricing error metric: √(e'M^(-1)e))

**Contents**: Performance metrics for portfolios of IPCA factors

---

### 5. `'info_list'` - Convergence Information

- **Type**: `list` of `dict`
- **Length**: `n_windows` (number of rolling windows)
- **Contents**: Each dict contains convergence info for one window:
  - `'iterations'`: Number of optimization iterations
  - `'converged'`: Whether optimization converged (True/False)
  - `'final_cost'`: Final objective function value
  - Additional optimization metadata from pymanopt

**Creation trace**:
1. [run_ipca.py:79-88](run_ipca.py#L79-L88): Returned from `ipca.fit_ipca_rolling()`
2. [ipca_functions.py:273-370](utils_factors/ipca_functions.py#L273-L370): Collected in `fit_ipca_rolling()`
   - For each window, stores info dict from `fit_ipca_single_window()`
3. [ipca_functions.py:230-268](utils_factors/ipca_functions.py#L230-L268): Created in `fit_ipca_single_window()`
   - Tracks Stiefel manifold optimization progress
   - Records convergence metrics

**Contents**: Diagnostic information about IPCA estimation quality

**Usage**: Check convergence and iteration counts
```python
# Average iterations across all windows
avg_iters = np.mean([info['iterations'] for info in info_list])

# Check if any windows failed to converge
failures = [i for i, info in enumerate(info_list) if not info.get('converged', True)]
```

---

### 6. `'K'` - Number of Latent Factors

- **Type**: `int`
- **Value**: Specified in command line (e.g., 1, 2, 3)

**Creation trace**:
- [run_ipca.py:171](run_ipca.py#L171): Parsed from command-line arguments

**Contents**: Number of latent factors (K parameter)

**Constraints**:
- Must satisfy K ≤ L where L = number of characteristics + 1
- Validated at [run_ipca.py:174-179](run_ipca.py#L174-L179)

---

### 7. `'panel_id'` - Panel Identifier

- **Type**: `str`
- **Value**: e.g., `'bgn_0'`, `'kp14_5'`, `'gs21_12'`

**Creation trace**:
- [run_ipca.py:156](run_ipca.py#L156): Parsed from command-line arguments via `factor_utils.parse_panel_arguments()`

**Contents**: Identifier linking this output to the source panel data

---

### 8. `'model'` - Model Name

- **Type**: `str`
- **Value**: `'bgn'`, `'kp14'`, or `'gs21'`

**Creation trace**:
- [run_ipca.py:182](run_ipca.py#L182): Extracted from CONFIG via `CONFIG['model']`

**Contents**: Model identifier

---

### 9. `'chars'` - Characteristic Names

- **Type**: `list`
- **Value**:
  - BGN/KP14: `['size', 'bm', 'agr', 'roe', 'mom']` (5 characteristics)
  - GS21: `['size', 'bm', 'agr', 'roe', 'mom', 'mkt_lev']` (6 characteristics)

**Creation trace**:
- [run_ipca.py:183](run_ipca.py#L183): Extracted from CONFIG via `CONFIG['chars']`
- Defined in [config.py:77-82](config.py#L77-L82)

**Contents**: List of firm characteristics used in IPCA estimation

---

### 10. `'start'` - Start Month

- **Type**: `int`
- **Value**: First month after burnin (e.g., 300 for burnin=300)

**Creation trace**:
- [run_ipca.py:189](run_ipca.py#L189): Returned from `factor_utils.prepare_panel()`

**Contents**: First month in panel (factor returns start at start+360)

---

### 11. `'end'` - End Month

- **Type**: `int`
- **Value**: Last month in panel (e.g., 699 for T=400, burnin=300)

**Creation trace**:
- [run_ipca.py:189](run_ipca.py#L189): Returned from `factor_utils.prepare_panel()`

**Contents**: Last month in panel

---

## Summary

The pickle file contains:
- **Factor returns** (1 DataFrame): IPCA latent factor returns
- **Loadings** (1 3D array): Factor loadings for each firm at each time
- **Portfolio weights** (1 array): Weights on factors for efficient portfolio
- **Statistics** (1 DataFrame): Performance metrics for portfolios of IPCA factors
- **Convergence info** (1 list): Optimization diagnostics for each window
- **Metadata** (5 items): K, panel_id, model, chars, start, end

**Key dependencies**:
- Requires `{panel_id}_panel.pkl` (panel data)
- Requires `{panel_id}_moments.pkl` (pre-computed SDF moments for statistics)
- Requires `pymanopt>=2.2.0` for Stiefel manifold optimization
- Uses 360-month rolling windows for estimation

## IPCA Methodology

Instrumented Principal Component Analysis (IPCA) estimates latent factors where loadings are linear functions of firm characteristics:

1. **Model**: `R_it = β_it' f_t + ε_it` where `β_it = X_it Gamma`
   - R_it: firm i's return at time t
   - X_it: firm i's characteristics at time t (L-vector)
   - Gamma: coefficient matrix (L × K)
   - f_t: latent factors at time t (K-vector)

2. **Estimation**:
   - Uses 360-month rolling windows
   - Optimizes on Stiefel manifold: Gamma such that Gamma'Gamma = I (orthogonality)
   - Minimizes sum of squared pricing errors
   - Uses pymanopt with conjugate gradient optimizer

3. **Identification**:
   - Factors are orthogonal (Gamma'Gamma = I)
   - Sign normalization applied (factors have positive mean)
   - Scale normalization via SVD

4. **Rolling estimation**:
   - Window [t-359, t]: estimate Gamma_t and f_t
   - Use Gamma_t and characteristics at t+360 to predict returns at t+360
   - Warm-start next window with previous Gamma_t

## Usage Example

```python
import pickle
import numpy as np
import pandas as pd

# Load the pickle file
with open('outputs/bgn_0_ipca_3.pkl', 'rb') as f:
    results = pickle.load(f)

# Access IPCA factor returns
ipca_factors = results['ipca_factors']  # (T-360, 3) for K=3

# Access factor loadings
ipca_weights = results['ipca_weights']  # (3, N, T-360)

# Get loadings for firm 10 at window 50
loadings_firm10_win50 = ipca_weights[:, 10, 50]  # (3,) vector

# Access portfolio weights
ipca_pi = results['ipca_pi']  # (3, T-360)

# Construct IPCA portfolio for window 50
portfolio_ret_50 = ipca_factors.iloc[50, :] @ ipca_pi[:, 50]

# Access convergence info
info_list = results['info_list']
avg_iterations = np.mean([info['iterations'] for info in info_list])
print(f"Average iterations per window: {avg_iterations:.1f}")

# Check for convergence issues
non_converged = [i for i, info in enumerate(info_list)
                 if not info.get('converged', True)]
if non_converged:
    print(f"Warning: {len(non_converged)} windows did not converge")

# Access metadata
K = results['K']  # Number of factors
panel_id = results['panel_id']
model = results['model']
chars = results['chars']
```

## Related Files

- **Input**: `outputs/{panel_id}_panel.pkl` - Panel data with characteristics
- **Input**: `outputs/{panel_id}_moments.pkl` - Pre-computed SDF moments
- **Script**: `run_ipca.py` - Generates this pickle file
- **Config**: `config.py` - Configuration parameters
  - `IPCA_K_VALUES`: List of K values to compute (default: [1, 2, 3])
  - `IPCA_N_RESTARTS`: Random restarts for first window (default: 3)
  - `IPCA_MAX_ITERATIONS`: Max optimization iterations (default: 100)
  - `IPCA_SIGN_NORMALIZE`: Apply sign normalization (default: True)
  - `IPCA_WARM_START`: Warm-start from previous window (default: True)

## Notes

- **Factor returns** start at month `start + 360` due to 360-month estimation window
- **Loadings array** is sparse - many zeros for firms that don't exist at certain times
- **Sign indeterminacy**: Factors are only identified up to sign (±1)
- **Optimization**: Uses Stiefel manifold to enforce orthogonality constraint
- **Convergence**: Check `info_list` for convergence issues
- **Statistics**: Computed using ipca_weights loadings and pre-computed SDF moments
- **Memory**: `ipca_weights` array can be large for high N (stored as float64)
- **Warm starting**: Uses previous window's Gamma as initial guess for faster convergence
- **pymanopt**: Required dependency for Stiefel manifold optimization

## Performance Considerations

- IPCA is computationally expensive compared to Fama or DKKM
- Complexity scales with:
  - K (number of factors) - higher K → more optimization variables
  - N (number of firms) - larger panels → larger matrices
  - Number of windows - longer time series → more estimations
- Typical runtime: 1-5 minutes per window depending on K and N
- Use warm starting (`IPCA_WARM_START=True`) to reduce runtime
- Monitor convergence via `info_list` to detect issues
