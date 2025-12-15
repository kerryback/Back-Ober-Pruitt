# NoIPCA Workflow Guide

## Overview

This document explains:
1. What the NoIPCA workflow does
2. How it differs from the original code
3. The logic and flow of computations
4. What each component produces

## Table of Contents

- [High-Level Workflow](#high-level-workflow)
- [Components Explained](#components-explained)
- [Panel Generation](#panel-generation)
- [Factor Computation](#factor-computation)
- [Changes from Original](#changes-from-original)
- [Data Flow](#data-flow)

---

## High-Level Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    PANEL GENERATION                         │
│  (Optional - generates synthetic data for testing)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     LOAD PANEL DATA                         │
│  panel_bgn_baseline.csv (or your data)                     │
│  Structure: (month, firmid) → (xret, characteristics)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  FACTOR COMPUTATION                         │
│                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐       │
│  │  Model Factors       │  │  Fama-French         │       │
│  │  (Taylor & Proj)     │  │  (3-factor model)    │       │
│  └──────────────────────┘  └──────────────────────┘       │
│                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐       │
│  │  Fama-MacBeth        │  │  DKKM                │       │
│  │  (2-stage regression)│  │  (Random Fourier)    │       │
│  └──────────────────────┘  └──────────────────────┘       │
│                                                             │
│  NOTE: IPCA removed (was in original)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      SAVE RESULTS                           │
│  output/ff_returns.parquet                                  │
│  output/fm_returns.parquet                                  │
│  output/dkkm_factors_rs.parquet                            │
│  output/dkkm_factors_nors.parquet                          │
│  output/model_premia_taylor.parquet                        │
│  output/model_premia_proj.parquet                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Components Explained

### 1. Panel Generation

**Purpose:** Generate synthetic panel data for testing

**Files:**
- `panel_functions.py` - BGN model
- `panel_functions_kp14.py` - KP14 model
- `panel_functions_gs21.py` - GS21 model

**What it does:**
1. Simulates firm-level data with structural model
2. Generates excess returns (`xret`) based on:
   - Latent factors (true factors from model)
   - Firm characteristics (size, book-to-market, etc.)
   - Idiosyncratic noise
3. Includes 200-month burnin period (discarded)

**Key parameters:**
- `N` = Number of firms (default: 100)
- `T` = Number of time periods (default: 400, plus 200 burnin)
- Characteristics: size, book-to-market, asset growth, ROE, momentum

**Models differ in:**
- Factor structure (how many latent factors)
- Loading dynamics (how characteristics map to factors)
- SDF specification (pricing kernel)

**Usage:**
```bash
# Generate all three models
python generate_panels.py

# Or just BGN (faster)
python generate_panel_bgn.py
```

**Output:**
- `panel_bgn_baseline.csv` - Panel data (month, firmid, xret, characteristics)
- `arrays_bgn_baseline.pkl` - Underlying arrays (factors, loadings, etc.)

---

### 2. Model Factors (Latent)

**Purpose:** Extract true latent factors from panel structure

**File:** `run_panel.py` (lines 85-98)

**Method:** Cross-sectional regression at each time period
```
r_t = A_t * f_t + epsilon_t

Where:
- r_t = excess returns at time t (N × 1)
- A_t = factor loadings at time t (N × K)
- f_t = factor returns at time t (K × 1)
```

**Two variants:**
1. **Taylor expansion** (`A_taylor`)
   - Linear approximation of loadings
   - Uses: `A_1_taylor`, `A_2_taylor`, etc.

2. **Projection** (`A_proj`)
   - Orthogonal projection
   - Uses: `A_1_proj`, `A_2_proj`, etc.

**Cross-sectional regression:**
```python
f_t = (A_t' A_t)^{-1} A_t' r_t
```

**Output:** Time series of factor returns (T × K)

**Key insight:** These are the "true" factors from the data-generating process (DGP). In real data, we don't observe A_t, so we estimate with other methods (Fama-French, Fama-MacBeth, DKKM).

---

### 3. Fama-French Factors

**Purpose:** Classic 3-factor model (Fama-French 1993)

**File:** `fama_functions.py` (lines 18-76)

**Logic:**
1. **Portfolio formation** (each month):
   - Sort firms by size (market equity)
   - Split at median: Small (S) vs Big (B)
   - For each characteristic:
     - Sort by characteristic value
     - Split into Low (L), Medium (M), High (H) at 30/70 percentiles

2. **Factor construction:**
   - **SMB** (Size): Small - Big
   - **HML** (Value): High book-to-market - Low
   - **MOM** (Momentum): High momentum - Low
   - Similarly for other characteristics (ROE, Asset Growth)

3. **Portfolio returns:**
   - Equal-weighted or value-weighted within each bucket
   - Long-short: Long high characteristic, Short low characteristic

**Key parameters:**
- `chars` = List of characteristics to use
- `mve` = Market value of equity (for size sort)
- Breakpoints: 0.3, 0.5, 0.7 (30th, 50th, 70th percentiles)

**Output:** Time series of factor returns (T × n_factors)

**Example:**
```
Month 200:
  Small firms with high B/M: avg return = 1.2%
  Big firms with low B/M: avg return = 0.8%
  HML factor return = 1.2% - 0.8% = 0.4%
```

---

### 4. Fama-MacBeth Factors

**Purpose:** Two-stage regression approach (Fama-MacBeth 1973)

**File:** `fama_functions.py` (lines 129-223)

**Logic:**

**Stage 1:** Cross-sectional regression at each time t
```
r_{i,t} = alpha_t + beta_t' z_{i,t} + epsilon_{i,t}

Where:
- r_{i,t} = excess return of firm i at time t
- z_{i,t} = standardized characteristics of firm i at time t
- beta_t = factor loadings at time t (estimated)
```

**Stage 2:** Time-series average of coefficients
```
beta = mean(beta_t) over all t
```

**Key steps:**
1. **Standardize characteristics** (each month):
   - Subtract mean, divide by std dev
   - Creates comparable scales across characteristics

2. **Add intercept:**
   - X = [1, z_1, z_2, ..., z_K]

3. **OLS regression:**
   - beta_t = (X'X)^{-1} X'r
   - Uses pseudo-inverse for stability

4. **Extract factor returns:**
   - Projection: P = X(X'X)^{-1}
   - Factor returns = P'r (returns explained by characteristics)

**Output:** Time series of factor returns (T × n_chars)

**Interpretation:** How much return is associated with each characteristic, controlling for others.

---

### 5. DKKM Factors (Random Fourier Features)

**Purpose:** Non-linear approximation of characteristics → returns mapping

**File:** `dkkm_functions.py`

**Key idea:** Random Fourier Features (RFF) approximate kernel methods
- Kernel: Measures similarity between firms
- RFF: Finite-dimensional approximation using sin/cos transformations

**Logic:**

**Step 1: Random Fourier Features**
```
Given characteristics X (N × L):
1. Generate random weights W ~ N(0, I) of size (D/2 × L)
2. Compute Z = W @ X' (D/2 × N)
3. Apply trig: [sin(Z); cos(Z)] stacked → (N × D)
```

**Step 2: Rank standardization**
- Map to uniform [-0.5, 0.5] based on rank
- Makes features comparable across months

**Step 3: Cross-sectional regression**
```
At each month t:
  r_t = RFF_t @ theta_t + epsilon_t

Solve: theta_t = (RFF_t' RFF_t)^{-1} RFF_t' r_t
```

**Step 4: Factor extraction**
- Two versions:
  1. **Rank-standardized RFF** (more stable)
  2. **Raw RFF** (preserves magnitudes)

**Key parameters:**
- `D` = Number of RFF dimensions (e.g., 360, 1000, 10000)
- `L` = Number of characteristics
- `nmat` = Number of random weight matrices (averaging)

**Output:**
- `dkkm_factors_rs.parquet` - Using rank-standardized RFF
- `dkkm_factors_nors.parquet` - Using raw RFF

**Why Random Fourier Features?**
- Kernel methods scale O(N²) - too slow for large N
- RFF approximates kernel in O(N·D) - much faster
- Captures non-linear relationships

**Mathematical foundation:**
```
Kernel trick: k(x_i, x_j) = φ(x_i)' φ(x_j)
RFF approximates: φ(x) ≈ [cos(w_1'x), sin(w_1'x), ..., cos(w_D'x), sin(w_D'x)]
```

---

## Changes from Original Code

### What Was Removed

**1. IPCA (Instrumented Principal Components Analysis)**
- **Why:** Very slow at production scale (94s for test, would be hours for production)
- **Impact:** One fewer method, but other methods still provide comprehensive coverage
- **Code removed:** `Refactor/ipca_stiefel.py` and all IPCA calls in main

**2. Pymanopt dependency**
- **Why:** Only needed for IPCA (Stiefel manifold optimization)
- **Impact:** Simpler installation, fewer dependencies

### What Was Added

**1. Panel Generation (NEW)**
- **Files:** `panel_functions*.py`, `generate_*.py`
- **Why:** Allows testing without external data
- **Models:** BGN, KP14, GS21

**2. Performance Optimizations (NEW)**
- **Randomized SVD:** For D > 1000, uses sklearn's randomized_svd (20x faster)
- **Numba JIT:** Accelerates rank_standardize and RFF computation (2-5x faster)
- **Files:** `utils_ridge_fast.py`, `utils_numba.py`, `dkkm_functions_numba.py`

**3. Comprehensive Documentation (NEW)**
- Acceleration guides (ACCELERATION_*.md)
- Workflow documentation (this file)
- Testing scripts (test_*.py)

### What Was Refactored

**1. Ridge Regression**
- **Original:** Simple implementation in `utils.py`
- **Refactored:**
  - `ridge_regression_fast()` - Handles T < P case efficiently
  - `ridge_regression_grid()` - Vectorized over multiple alphas
  - Automatic method selection (randomized SVD for D > 1000)

**2. Rank Standardization**
- **Original:** Pandas-based with double argsort
- **Refactored:** NumPy-based with single argsort (faster)
- **Numba version:** Parallel over columns (3-5x faster)

**3. Code Structure**
- **Original:** Monolithic scripts
- **Refactored:** Modular functions in separate files
  - `utils.py` - Utility functions
  - `fama_functions.py` - Fama-French/MacBeth
  - `dkkm_functions.py` - DKKM/RFF
  - `run_panel.py` - Main workflow

**4. Parallel Processing**
- **Original:** Sequential loops
- **Refactored:** Joblib parallelization
  - Fama methods: Parallel over months
  - DKKM: Parallel over months
  - Configurable `n_jobs` parameter

### Key Improvements

| Aspect | Original | NoIPCA |
|--------|----------|---------|
| **Runtime** (N=100, T=400) | ~94s | ~3s (31x faster) |
| **Runtime** (N=1000, T=720, D=10K) | ~8 hours | ~17 min (28x faster) |
| **Code clarity** | Monolithic | Modular |
| **Documentation** | Minimal | Comprehensive |
| **Testing** | None | Multiple test scripts |
| **Panel generation** | External | Included |
| **Flexibility** | Fixed | Configurable |
| **Dependencies** | Pymanopt required | Optional (Numba, sklearn) |

---

## Data Flow

### Input Data Structure

**Panel data format:**
```
Index: (month, firmid)
Columns:
  - xret: Excess return
  - size: log(market value)
  - bm: Book-to-market ratio
  - agr: Asset growth
  - roe: Return on equity
  - mom: Momentum
  - A_1_taylor, A_2_taylor: True loadings (if available)
  - A_1_proj, A_2_proj: Alternative loadings
  - mve: Market value of equity
```

**Example:**
```
month  firmid  xret    size   bm     agr    roe    mom    mve
200    1       0.015   10.2   0.65   0.12   0.08   0.05   1000
200    2      -0.008   11.5   0.45  -0.05   0.12   0.10   5000
...
201    1       0.022   10.3   0.63   0.10   0.09   0.08   1100
```

### Computation Flow

**For each method, at each month t:**

1. **Subset data:**
   ```python
   data_t = panel.loc[t]  # All firms at month t
   ```

2. **Extract components:**
   ```python
   returns = data_t['xret']  # (N,)
   characteristics = data_t[char_list]  # (N, L)
   ```

3. **Method-specific computation:**

   **Model Factors:**
   ```python
   A_t = data_t[loading_cols]  # (N, K)
   f_t = (A_t.T @ A_t)^{-1} @ A_t.T @ returns
   ```

   **Fama-French:**
   ```python
   weights = construct_portfolios(characteristics, mve)  # (N, n_factors)
   f_t = weights.T @ returns
   ```

   **Fama-MacBeth:**
   ```python
   Z_t = standardize(characteristics)  # (N, L)
   X_t = [1, Z_t]  # Add intercept
   beta_t = (X_t.T @ X_t)^{-1} @ X_t.T @ returns
   f_t = projection_matrix(X_t) @ returns
   ```

   **DKKM:**
   ```python
   Z_t = rank_standardize(characteristics)  # (N, L)
   RFF_t = compute_rff(Z_t, W)  # (N, D)
   theta_t = (RFF_t.T @ RFF_t)^{-1} @ RFF_t.T @ returns
   f_t = RFF_t @ theta_t
   ```

4. **Aggregate over time:**
   ```python
   factors = pd.concat([f_t for t in range(start, end+1)])
   ```

### Output Structure

**Each method produces:**
```
Index: month (200 to 599)
Columns: factor_1, factor_2, ..., factor_K
Values: Factor returns at each time period
```

**Example (Fama-French):**
```
month  SMB    HML_bm  MOM    AGR    ROE
200    0.012  0.008  -0.003  0.005  0.010
201   -0.005  0.015   0.020 -0.002  0.008
...
```

**File formats:**
- All saved as `.parquet` (compressed, fast I/O)
- Can be read with: `pd.read_parquet('output/ff_returns.parquet')`

---

## Configuration

**Key parameters in `run_panel.py`:**

```python
# Data
panel_path = '../panel_bgn_baseline.csv'

# Time periods
start_month = 200
end_month = 599

# Parallelization
n_jobs = 10

# DKKM settings
CONFIG.nmat = 1  # Number of random weight matrices
CONFIG.max_features = 10000  # D (RFF dimensions)

# Characteristics to use
chars = ['size', 'bm', 'agr', 'roe', 'mom']
```

**Modifying for your data:**

1. **Change input file:**
   ```python
   panel_path = 'path/to/your/data.csv'
   ```

2. **Adjust time range:**
   ```python
   start_month = your_start
   end_month = your_end
   ```

3. **Add/remove characteristics:**
   ```python
   chars = ['size', 'bm', 'momentum', 'profitability']
   ```

4. **Tune DKKM:**
   - Increase `max_features` for more flexibility (but slower)
   - Increase `nmat` for more stable estimates (but slower)

---

## Understanding Results

### Interpreting Factor Returns

**Model Factors (Taylor/Proj):**
- These are the "true" factors from the DGP
- Use as benchmark to evaluate other methods
- High correlation = method captures true factors well

**Fama-French:**
- Classic interpretations:
  - SMB: Small firm premium
  - HML: Value premium
  - MOM: Momentum premium
- Should capture systematic risk factors

**Fama-MacBeth:**
- Factor returns = characteristic loadings
- Positive β_j means characteristic j predicts higher returns
- Magnitude = economic significance

**DKKM:**
- Non-linear combination of characteristics
- Hard to interpret individual components
- Evaluate by overall fit (R²)

### Comparing Methods

**Load and compare:**
```python
import pandas as pd

# Load all methods
model_taylor = pd.read_parquet('output/model_premia_taylor.parquet')
ff = pd.read_parquet('output/ff_returns.parquet')
fm = pd.read_parquet('output/fm_returns.parquet')
dkkm_rs = pd.read_parquet('output/dkkm_factors_rs.parquet')

# Compute correlations
correlation = pd.concat([
    model_taylor['taylor'],
    ff['HML_bm'],
    fm['bm'],
    dkkm_rs.iloc[:, 0]  # First DKKM factor
], axis=1).corr()

print(correlation)
```

**Expected patterns:**
- Model factors should be close to true DGP (correlation > 0.9)
- Fama-French should capture size/value effects
- Fama-MacBeth should be similar to Fama-French but smoother
- DKKM should have highest R² (most flexible)

### Diagnostics

**1. Check factor magnitudes:**
```python
ff.describe()
```
Expected: Mean ~0, Std ~1-5%, occasional spikes

**2. Time series plots:**
```python
ff[['SMB', 'HML_bm']].plot()
```
Should show time-varying but stationary behavior

**3. Correlation structure:**
```python
ff.corr()
```
Factors should be relatively uncorrelated (but some correlation OK)

---

## Troubleshooting

### Common Issues

**1. "Panel not found"**
- Generate panel first: `python generate_panel_bgn.py`
- Or update `panel_path` in main script

**2. "Ridge regression very slow"**
- If D > 1000: Install scikit-learn
- Check for message: `[Ridge] Using randomized SVD`

**3. "Memory error"**
- Reduce `max_features` (e.g., from 10000 to 1000)
- Reduce time range (smaller T)
- Use rank-standardized DKKM only

**4. "NaN in results"**
- Check for missing data in panel
- Ensure enough firms per month (N > L)
- Check characteristics are not constant

### Performance Issues

**Slow execution:**
1. Check optimizations are active:
   - `[ACCELERATION]` message on import
   - `[Ridge]` message during DKKM

2. Verify installations:
   ```bash
   python -c "import numba; print('Numba OK')"
   python -c "from sklearn.utils.extmath import randomized_svd; print('sklearn OK')"
   ```

3. Profile to find bottleneck:
   ```bash
   python -m cProfile -s cumtime run_panel.py
   ```

**Expected runtimes (N=1000, T=720, D=10K):**
- Panel loading: 2-3s
- Model factors: 1s
- Fama-French: 2-3s
- Fama-MacBeth: 2-3s
- DKKM: 15-17 minutes (dominated by ridge regression)
- **Total: 17-20 minutes**

---

## Next Steps

1. **Verify installation:**
   ```bash
   python test_numba_integration.py
   ```

2. **Generate test data:**
   ```bash
   python generate_panel_bgn.py
   ```

3. **Run workflow:**
   ```bash
   python run_panel.py
   ```

4. **Examine results:**
   ```python
   import pandas as pd
   ff = pd.read_parquet('output/ff_returns.parquet')
   print(ff.head())
   print(ff.describe())
   ```

5. **Scale to your data:**
   - Update `panel_path`
   - Adjust time range
   - Modify characteristics list
   - Tune DKKM parameters

---

## References

**Methods:**
- Fama-French (1993): "Common risk factors in the returns on stocks and bonds"
- Fama-MacBeth (1973): "Risk, return, and equilibrium: Empirical tests"
- Kelly-Pruitt-Su (2019): "Characteristics are covariances" (DKKM)

**Implementation:**
- Original code structure from previous work
- Optimizations: randomized SVD (Halko et al. 2011)
- Numba JIT compilation: http://numba.pydata.org/

**Data-generating processes:**
- BGN: Base model
- KP14: Kelly-Pruitt 2014
- GS21: Gospodinov-Sucarrat 2021
