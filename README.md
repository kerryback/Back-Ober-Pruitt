# Back-Ober-Pruitt: Asset Pricing Factor Models

A comprehensive Python framework for simulating asset pricing models and computing risk factors using multiple methodologies. This project implements three structural asset pricing models (BGN, KP14, GS21) and evaluates factor models using Fama-French, DKKM, and IPCA approaches.

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Factor Methodologies](#factor-methodologies)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Requirements](#requirements)

---

## Overview

This project evaluates different factor model estimation techniques by:

1. **Simulating panel data** from structural asset pricing models with known SDF moments
2. **Computing risk factors** using multiple methodologies (Fama-French, DKKM, IPCA)
3. **Evaluating factor performance** via portfolio statistics and Sharpe ratios

The key insight is that we can test factor models against a **known truth** (the structural model's true SDF), rather than relying solely on empirical data.

---

## Models

### BGN Model
Based on **Berk, Green, and Naik (1999)** - a real options model with:
- **Characteristics**: size, book-to-market (bm), asset growth (agr), return on equity (roe), momentum (mom)
- **True Factors**: 2 latent factors
- **Features**: Firms as growth options with endogenous investment decisions

### KP14 Model
Based on **Kuehn and Petrosky-Nadeau (2014)** - a production economy with:
- **Characteristics**: size, bm, agr, roe, mom
- **True Factors**: 2 latent factors
- **Features**: Financial frictions, endogenous default, and time-varying risk premia

### GS21 Model
Based on **Gomes and Schmid (2021)** - an extended production model with:
- **Characteristics**: size, bm, agr, roe, mom, market leverage (mkt_lev)
- **True Factors**: 1 latent factor
- **Features**: Corporate defaults, financing frictions, and capital structure decisions

---

## Factor Methodologies

### 1. Fama-French (FF)
**Characteristic-sorted portfolios** - the classic approach:
- Sort firms by characteristics (e.g., size, value, profitability)
- Form portfolios and compute returns
- **Pros**: Interpretable, easy to implement
- **Cons**: Requires arbitrary sorting decisions

### 2. DKKM (Random Fourier Features)
**Data-driven kernel approximation** using random features:
- Approximate SDF using Random Fourier Features (RFF)
- Flexible nonlinear relationships
- **Pros**: Minimal assumptions, data-driven
- **Cons**: High dimensionality, requires careful tuning

### 3. IPCA (Instrumented PCA)
**Latent factor model** with characteristic loadings:
- Estimate latent factors via Stiefel manifold optimization
- Factor loadings are linear functions of characteristics
- **Pros**: Theoretically grounded, efficient
- **Cons**: Requires pymanopt, computationally intensive

---

## Installation

### Prerequisites
- Python 3.8+
- Windows (code includes Windows-specific encoding handling)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kerryback/Back-Ober-Pruitt.git
   cd Back-Ober-Pruitt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Key dependencies**:
   - `numpy`, `pandas`, `scipy` - Numerical computing
   - `statsmodels` - Statistical analysis
   - `joblib` - Parallel processing
   - `pymanopt>=2.2.0` - IPCA optimization (Riemannian manifolds)

---

## Quick Start

### Run Complete Workflow

The simplest way to run the entire workflow:

```bash
# Generate all outputs for KP14 model, indices 0-4
python main.py kp14 0 5
```

This single command executes all 5 steps for each index:
1. Generate panel data
2. Calculate SDF moments
3. Compute Fama-French factors
4. Compute DKKM factors (for all configured feature counts)
5. Compute IPCA factors (for all configured K values)

### Individual Steps

You can also run each step independently:

```bash
# Step 1: Generate panel data
python generate_panel.py kp14 0

# Step 2: Calculate SDF moments (expensive computation)
python calculate_moments.py kp14_0

# Step 3: Fama-French factors
python run_fama.py kp14_0

# Step 4: DKKM factors (specify number of features)
python run_dkkm.py kp14_0 360

# Step 5: IPCA factors (specify number of latent factors K)
python run_ipca.py kp14_0 3
```

---

## Workflow

### Architecture

The workflow follows a **modular pipeline**:

```
┌─────────────────────┐
│  1. Generate Panel  │  <- Simulate data from structural model
│   (BGN/KP14/GS21)   │
└──────────┬──────────┘
           │ Output: {model}_{id}_arrays.pkl
           ▼
┌─────────────────────┐
│ 2. Calculate SDF    │  <- Compute true SDF moments (expensive)
│     Moments         │
└──────────┬──────────┘
           │ Output: {model}_{id}_moments.pkl
           ▼
   ┌───────┴───────┬──────────────┐
   │               │              │
   ▼               ▼              ▼
┌──────┐      ┌───────┐      ┌───────┐
│ Fama │      │ DKKM  │      │ IPCA  │  <- Estimate factors
└───┬──┘      └───┬───┘      └───┬───┘
    │             │              │
    │             │              │
    └─────────────┴──────────────┘
                  │
                  ▼ Output: {model}_{id}_{method}.pkl
          ┌──────────────┐
          │   Analysis   │  <- Evaluate performance
          │  (Optional)  │
          └──────────────┘
```

### Step Details

#### Step 1: Generate Panel (`generate_panel.py`)
- **Input**: Model name (bgn/kp14/gs21), identifier
- **Process**:
  - Solve model for firm value functions and SDFs
  - Simulate firm dynamics (returns, book values, characteristics)
  - Generate panel with N firms × T months
- **Output**: `{model}_{id}_arrays.pkl` containing:
  - `panel`: DataFrame with returns and characteristics
  - `A_*_taylor`, `A_*_proj`: True SDF loadings (Taylor vs projection)
  - `f_*_taylor`, `f_*_proj`: True factors

#### Step 2: Calculate Moments (`calculate_moments.py`)
- **Input**: Panel data from Step 1
- **Process**:
  - Compute SDF conditional moments for each month
  - Use **parallel processing** (joblib) for speed
  - Process in chunks to manage memory (important for large panels)
- **Output**: `{model}_{id}_moments.pkl` containing:
  - `rp`: Risk premia (N × T)
  - `cond_var`: Conditional variance (N × T)
  - `second_moment`: E[mm'] (N × N × T)
  - `second_moment_inv`: [E[mm']]^{-1} (N × N × T)

**Why separate from factor computation?**
These moments are computationally expensive and reused across all factor methods, so we compute once and cache.

#### Step 3-5: Factor Computation
Each method reads both `arrays.pkl` and `moments.pkl`, then:
- Estimates factors using its methodology
- Constructs portfolios with ridge regularization
- Evaluates performance (Sharpe ratios, alphas, R²)
- Saves results to `{model}_{id}_{method}.pkl`

---

## Configuration

All parameters are centralized in [`config.py`](config.py):

### Panel Dimensions
```python
N = 1000        # Number of firms
T = 720         # Time periods (months, excluding burnin)
BGN_BURNIN = 200   # Burnin period for BGN
KP14_BURNIN = 200  # Burnin period for KP14
GS21_BURNIN = 200  # Burnin period for GS21
N_JOBS = 10     # Parallel jobs for moment computation
```

### DKKM Parameters
```python
N_DKKM_FEATURES_LIST = [6, 36, 360]  # RFF feature counts to test
DKKM_RANK_STANDARDIZE = True         # Rank-standardize characteristics
INCLUDE_MKT = True                   # Include market portfolio
```

### IPCA Parameters
```python
IPCA_K_VALUES = [1, 2, 3]           # Number of latent factors
IPCA_N_RESTARTS = 3                 # Random restarts for robustness
IPCA_MAX_ITERATIONS = 100           # Max iterations for optimization
IPCA_SIGN_NORMALIZE = True          # Normalize factor signs
IPCA_WARM_START = True              # Use previous solution as initial guess
```

### Ridge Regularization
```python
# Shrinkage penalties (Berk-Jameson regularization)
ALPHA_LST = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]  # For BGN/KP14
ALPHA_LST_GS = [0, 0.0000001, ..., 0.1, 1]          # For GS21
IPCA_ALPHA_LST = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]
```

### Model Parameters

Each model has calibrated parameters (lines 85-159 in `config.py`):
- **BGN**: Persistence (PI), interest rate volatility (SIGMA_R), etc.
- **KP14**: Production parameters, default thresholds, risk aversion
- **GS21**: Financing frictions, tax rates, issuance costs

---

## Output Files

All outputs are saved to the `outputs/` directory:

### Generated Files (per panel)

| File Pattern | Description |
|-------------|-------------|
| `{model}_{id}_arrays.pkl` | Panel data + true SDF loadings/factors |
| `{model}_{id}_moments.pkl` | Cached SDF conditional moments |
| `{model}_{id}_fama.pkl` | Fama-French + Fama-MacBeth factors |
| `{model}_{id}_dkkm_{nfeatures}.pkl` | DKKM factors with specified features |
| `{model}_{id}_ipca_{K}.pkl` | IPCA factors with K latent factors |

### Pickle File Contents

Each factor output file contains:
```python
{
    'factors': np.ndarray,           # Factor returns (T × K)
    'loadings': np.ndarray,          # Factor loadings (N × K × T)
    'stats': dict,                   # Performance statistics
    'config': dict,                  # Configuration used
    'panel_id': str,                 # Panel identifier
    # ... method-specific fields
}
```

### Logs

All runs are logged to `logs/log_{model}_{start}_{end}.txt` with:
- Execution times for each step
- Configuration used
- Convergence diagnostics (for IPCA)
- Memory usage and performance metrics

---

## Project Structure

```
Back-Ober-Pruitt/
│
├── main.py                    # Master workflow orchestrator
├── generate_panel.py          # Step 1: Panel generation
├── calculate_moments.py       # Step 2: SDF moments
├── run_fama.py               # Step 3: Fama factors
├── run_dkkm.py               # Step 4: DKKM factors
├── run_ipca.py               # Step 5: IPCA factors
├── config.py                 # Centralized configuration
├── requirements.txt          # Python dependencies
│
├── utils_bgn/                # BGN model implementation
│   ├── panel_functions_bgn.py    # Panel generation
│   ├── sdf_compute_bgn.py        # SDF computation
│   ├── loadings_compute_bgn.py   # True loadings
│   └── vasicek.py                # Interest rate dynamics
│
├── utils_kp14/               # KP14 model implementation
│   ├── panel_functions_kp14.py
│   ├── sdf_compute_kp14.py
│   └── loadings_compute_kp14.py
│
├── utils_gs21/               # GS21 model implementation
│   ├── panel_functions_gs21.py
│   ├── sdf_compute_gs21.py
│   └── loadings_compute_gs21.py
│
├── utils_factors/            # Factor computation utilities
│   ├── fama_functions.py         # Fama-French/MacBeth
│   ├── dkkm_functions.py         # DKKM RFF implementation
│   ├── ipca_functions.py         # IPCA estimation
│   ├── portfolio_stats.py        # Performance evaluation
│   ├── ridge_utils.py            # Ridge regression utilities
│   └── factor_utils.py           # Common utilities
│
├── outputs/                  # Generated data files
├── logs/                     # Execution logs
├── tests/                    # Unit and workflow tests
└── .claude/                  # Claude Code configuration
```

---

## Technical Details

### Key Design Decisions

#### 1. ROE Calculation with Zero Book Equity
**Issue**: Firms with zero book equity cause `roe = NaN`, leading to data loss.

**Solution**: Set `roe = 0` when `book <= 0`:
```python
df["roe"] = df.groupby("firmid", group_keys=False).apply(
    lambda d: pd.Series(
        np.where(d.book > 0, d.op_cash_flow / d.book, 0),
        index=d.index
    ).shift()
)
```

Applied to all three models (BGN, KP14, GS21).

#### 2. Parallel Processing Strategy
- **Panel generation**: Sequential (model solving requires state)
- **Moment computation**: Parallel across months (`joblib` with `n_jobs=10`)
- **Factor computation**: Sequential for each panel, but can run multiple panels in parallel

#### 3. Memory Management
- `calculate_moments.py` processes in chunks (50 months at a time)
- Garbage collection (`gc.collect()`) between chunks
- Large arrays stored as `float64` only when necessary

#### 4. IPCA Optimization
- Uses **Stiefel manifold optimization** (pymanopt)
- Warm-start from previous window for speed
- Multiple random restarts for first window (robustness)
- Sign normalization for interpretability

#### 5. Windows-Specific Handling
**Console Encoding**: Windows uses `cp1252` encoding, which cannot handle Unicode box-drawing characters.

**Solution**: Use only ASCII characters in print statements:
```python
# Bad (causes UnicodeEncodeError on Windows)
print(f"{'─'*50}")

# Good (ASCII-safe)
print(f"{'-'*50}")
```

See [`.claude/CLAUDE.md`](.claude/CLAUDE.md) for complete Windows guidelines.

---

## Performance Considerations

### Computational Costs (Approximate, for N=1000, T=720)

| Step | Time | Bottleneck |
|------|------|-----------|
| Generate Panel | ~5-10 min | Model solving |
| Calculate Moments | ~30-60 min | SDF evaluation (parallel) |
| Fama Factors | ~2-5 min | Portfolio formation |
| DKKM (360 features) | ~10-20 min | Ridge regression |
| IPCA (K=3) | ~15-30 min | Stiefel optimization |

**Total per panel**: ~1-2 hours

### Optimization Tips

1. **Increase `N_JOBS`** if you have more cores (currently 10)
2. **Use smaller panels** for testing (e.g., N=100, T=400)
3. **Run multiple panels in parallel** using separate terminals
4. **Enable numba** (uncomment in `requirements.txt`) for ~2x speedup on ridge regression

---

## Example Usage Scenarios

### Scenario 1: Quick Test
```bash
# Small panel for testing
# Edit config.py: N = 100, T = 400
python main.py kp14 0 1
```

### Scenario 2: Single Model, Multiple Realizations
```bash
# Generate 10 independent panels for KP14
python main.py kp14 0 10
```

### Scenario 3: Compare Across Models
```bash
# Run all three models with same identifier
python main.py bgn 0 1
python main.py kp14 0 1
python main.py gs21 0 1
```

### Scenario 4: Sensitivity Analysis
```bash
# Test different DKKM feature counts
# Edit config.py: N_DKKM_FEATURES_LIST = [10, 50, 100, 500, 1000]
python main.py kp14 0 1
```

### Scenario 5: Custom Factor Configuration
```bash
# Run only specific factor methods
python generate_panel.py kp14 0
python calculate_moments.py kp14_0
python run_ipca.py kp14_0 5  # Test K=5 latent factors
```

---

## Troubleshooting

### Common Issues

**1. `UnicodeEncodeError` on Windows**
- **Cause**: Unicode characters in print statements
- **Fix**: Use only ASCII characters (see Windows guidelines above)

**2. `MemoryError` during moment calculation**
- **Cause**: Large panel dimensions (N × T too big)
- **Fix**: Reduce `N` or `T` in `config.py`, or increase chunk size

**3. IPCA fails to converge**
- **Cause**: Poor initialization or max iterations too low
- **Fix**: Increase `IPCA_MAX_ITERATIONS` or `IPCA_N_RESTARTS`

**4. `pymanopt` import error**
- **Cause**: Missing IPCA dependency
- **Fix**: `pip install pymanopt>=2.2.0`

**5. Negative Sharpe ratios**
- **Expected**: Some methods may underperform, especially with misspecification
- **Check**: Review `stats` dict in output files for diagnostics

---

## Requirements

See [`requirements.txt`](requirements.txt) for full list. Key dependencies:

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
statsmodels>=0.13.0
joblib>=1.1.0
pymanopt>=2.2.0
```

**Optional**:
- `numba>=0.56.0` - For accelerated ridge regression

**Python Version**: 3.8+

**Platform**: Windows (with some Unix compatibility)

---

## References

### Papers

1. **Berk, Green, and Naik (1999)**: "Optimal Investment, Growth Options, and Security Returns," *Journal of Finance*
2. **Kuehn and Petrosky-Nadeau (2014)**: "Financial Frictions and the Cross-Section of Expected Returns," Working Paper
3. **Gomes and Schmid (2021)**: "Equilibrium Asset Pricing with Leverage and Default," *Journal of Finance*
4. **Kelly, Pruitt, and Su (2019)**: "Characteristics Are Covariances: A Unified Model of Risk and Return," *Journal of Financial Economics* (DKKM method)
5. **Kelly, Pruitt, and Su (2019)**: "Instrumented Principal Component Analysis," Working Paper (IPCA method)

### Methodology

This codebase implements the simulation-based factor evaluation framework described in:
- **Back, Ober, and Pruitt (2024)**: "Evaluating Factor Models with Simulated Data," Working Paper

---

## License

This project is licensed under the MIT License.

## Contact

For questions or issues:
- **GitHub**: https://github.com/kerryback/Back-Ober-Pruitt
- **Issues**: https://github.com/kerryback/Back-Ober-Pruitt/issues

---

## Acknowledgments

- Simulation code based on original implementations by Kerry Back
- IPCA implementation adapted from Kelly, Pruitt, and Su (2019)
- Ridge regression utilities optimized for large-scale panel data

---

**Last Updated**: December 2024
