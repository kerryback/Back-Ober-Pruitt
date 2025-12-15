# Factor Computation Utilities (`utils_factors/`)

This directory contains specialized utilities for computing factor returns using different methodologies (Fama-French, Fama-MacBeth, DKKM) and analyzing portfolio statistics.

## Directory Contents

```
utils_factors/
├── __init__.py              # Package init
├── factor_utils.py          # Shared utilities (loading, preparing, saving data)
├── dkkm_functions.py        # DKKM (Random Fourier Features) factor computation
├── dkkm_functions_numba.py  # Numba-accelerated DKKM functions
├── fama_functions.py        # Fama-French & Fama-MacBeth factor computation
└── portfolio_stats.py       # Portfolio statistics computation
```

## Module Descriptions

### `factor_utils.py`
Shared utilities for factor computation scripts (run_fama.py, run_dkkm.py).

**Functions Exported:**
- `parse_panel_arguments(script_name, additional_args)` - Parse command-line arguments
- `load_model_config(model_name)` - Load model-specific configuration
- `load_panel_data(panel_id, model_name)` - Load panel data from pickle files
- `prepare_panel(panel, chars)` - Clean and prepare panel for factor computation
- `save_factor_results(results, output_file, verbose)` - Save results with summary
- `print_script_header(title, model, panel_id, config, additional_info)` - Print formatted header
- `print_script_footer(panel_id, usage_examples)` - Print footer with usage examples

**External Dependencies:**
- `numpy`, `pandas` - Data structures
- `pickle` - Serialization
- `pathlib`, `os`, `sys` - File operations
- `config` (BGN_CONFIG, KP14_CONFIG, GS21_CONFIG, DATA_DIR)

**Files Read:**
- `{DATA_DIR}/arrays_{panel_id}.pkl` - Panel data with characteristics and returns
- `{DATA_DIR}/panel_{panel_id}.pkl` - Alternative panel data format

**Files Written:**
- Output pickle files via `save_factor_results()`

### `dkkm_functions.py`
DKKM (Random Fourier Features) factor computation using kernel methods.

**Functions Exported:**
- `rff(data, rf, W, model)` - Compute Random Fourier Features
- `factors(panel, W, n_jobs, start, end, model, chars)` - Compute DKKM factor returns
- `mve_data(f, month, alpha_lst, mkt_rf)` - Mean-variance efficient portfolios

**External Dependencies:**
- `numpy`, `pandas` - Data structures
- `joblib.Parallel, delayed` - Parallel processing
- **From `utils/`:**
  - `from utils.numba import rank_standardize` (Numba-accelerated)
  - `from utils import ridge_regression_grid`
  - `from utils import rank_standardize` (fallback)

**Internal Imports:**
- `from .dkkm_functions_numba import rff_compute_numba` - Numba-accelerated RFF computation

**Performance:**
- Uses Numba acceleration when available (2-3x faster RFF computation)
- Parallel processing via joblib for monthly computations

### `dkkm_functions_numba.py`
Numba-accelerated Random Fourier Features computation.

**Functions Exported:**
- `rff_compute_numba(W, X)` - JIT-compiled RFF computation (2-3x faster)

**External Dependencies:**
- `numpy` - Array operations
- `numba` - JIT compilation

### `fama_functions.py`
Fama-French and Fama-MacBeth factor computation.

**Functions Exported:**
- `fama_french(data, chars, mve, **kwargs)` - Fama-French characteristic-sorted portfolios
- `fama_macbeth(data, chars, stdz_fm, **kwargs)` - Fama-MacBeth two-stage regression
- `factors(method, panel, n_jobs, start, end, chars, **kwargs)` - Unified factor computation

**External Dependencies:**
- `numpy`, `pandas` - Data structures
- `joblib.Parallel, delayed` - Parallel processing
- `scipy.linalg` - Linear algebra
- **From `utils/`:**
  - `from utils import ridge_regression_fast, standardize_columns`

**Performance:**
- Parallel processing via joblib for monthly computations
- Vectorized operations where possible

### `portfolio_stats.py`
Portfolio statistics computation for factor performance evaluation.

**Functions Exported:**
- `load_precomputed_moments(panel_id)` - Load pre-computed SDF moments
- `compute_model_portfolio_stats(model_premia, panel, start, end)` - Stats for model factors
- `compute_fama_portfolio_stats(ff_rets, fm_rets, panel, ...)` - Stats for Fama factors
- `compute_dkkm_portfolio_stats(dkkm_factors_rs, panel, W, ...)` - Stats for DKKM factors

**External Dependencies:**
- `numpy`, `pandas` - Data structures
- `pickle` - Serialization
- `config` (ModelConfig, DATA_DIR)

**Internal Imports:**
- `from . import fama_functions as fama`
- `from . import dkkm_functions as dkkm`

**Files Read:**
- `{DATA_DIR}/moments_{panel_id}.pkl` - Pre-computed SDF conditional moments

**Statistics Computed:**
- `stdev` - Standard deviation of factor returns
- `mean` - Mean factor returns
- `xret` - Sharpe ratio (mean/stdev)
- `hjd` - Hansen-Jagannathan distance

### `__init__.py`
Package initialization.

**Modules Exported:**
```python
from . import factor_utils
from . import dkkm_functions
from . import dkkm_functions_numba
from . import fama_functions
from . import portfolio_stats
```

## Usage Examples

### Loading Panel Data
```python
from utils_factors import factor_utils

# Parse arguments
panel_id, model_name, args = factor_utils.parse_panel_arguments('run_fama')

# Load configuration and data
CONFIG = factor_utils.load_model_config(model_name)
panel, arrays_data = factor_utils.load_panel_data(panel_id, model_name)
panel, start, end = factor_utils.prepare_panel(panel, CONFIG['chars'])
```

### Computing Fama Factors
```python
from utils_factors import fama_functions as fama

# Fama-French factors
ff_rets = fama.factors(fama.fama_french, panel,
                       n_jobs=10, start=200, end=600,
                       chars=["size", "bm", "agr", "roe", "mom"])

# Fama-MacBeth factors
fm_rets = fama.factors(fama.fama_macbeth, panel,
                       n_jobs=10, start=200, end=600,
                       chars=["size", "bm", "agr", "roe", "mom"],
                       stdz_fm=False)
```

### Computing DKKM Factors
```python
import numpy as np
from utils_factors import dkkm_functions as dkkm

# Generate Random Fourier Features weight matrix
W = np.random.normal(size=(180, 5))  # D/2=180, L=5 characteristics
gamma = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], size=(180, 1))
W = gamma * W

# Compute DKKM factors
dkkm_rs, dkkm_nors = dkkm.factors(panel, W, n_jobs=10,
                                   start=200, end=600,
                                   model='bgn',
                                   chars=["size", "bm", "agr", "roe", "mom"])
```

### Computing Portfolio Statistics
```python
from utils_factors import portfolio_stats

# For Fama factors
fama_stats = portfolio_stats.compute_fama_portfolio_stats(
    ff_rets, fm_rets, panel,
    panel_id='bgn_0',
    model='bgn',
    chars=["size", "bm", "agr", "roe", "mom"],
    start_month=200,
    end_month=600,
    alpha_lst=[0],
    burnin=200
)

# For DKKM factors
dkkm_stats = portfolio_stats.compute_dkkm_portfolio_stats(
    dkkm_factors_rs, panel,
    panel_id='bgn_0',
    model='bgn',
    W=W,
    chars=["size", "bm", "agr", "roe", "mom"],
    start_month=200,
    end_month=600,
    alpha_lst=[0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1],
    include_mkt=False,
    mkt_returns=None,
    matrix_idx=0,
    burnin=200
)
```

### Saving Results
```python
from utils_factors import factor_utils

results = {
    'ff_returns': ff_rets,
    'fm_returns': fm_rets,
    'fama_stats': fama_stats,
    'panel_id': 'bgn_0',
    'model': 'bgn',
    'chars': ["size", "bm", "agr", "roe", "mom"],
    'start': 200,
    'end': 600,
}

output_file = f"{DATA_DIR}/bgn_0_fama.pkl"
factor_utils.save_factor_results(results, output_file, verbose=True)
```

## External Modules That Import From `utils_factors/`

### Main Workflow Scripts

**`run_fama.py`:**
```python
from utils_factors import fama_functions as fama
from utils_factors import portfolio_stats
from utils_factors import factor_utils
```

**`run_dkkm.py`:**
```python
from utils_factors import dkkm_functions as dkkm
from utils_factors import portfolio_stats
from utils_factors import factor_utils
```

## Internal Cross-References

### Within `utils_factors/` directory:

**`portfolio_stats.py` imports:**
```python
from . import fama_functions as fama
from . import dkkm_functions as dkkm
```

**`dkkm_functions.py` imports:**
```python
from .dkkm_functions_numba import rff_compute_numba
```

### Imports from other NoIPCA packages:

**All modules import from `config`:**
```python
from config import ModelConfig, DATA_DIR
from config import BGN_CONFIG, KP14_CONFIG, GS21_CONFIG
```

**All factor computation modules import from `utils`:**
```python
# dkkm_functions.py
from utils.numba import rank_standardize
from utils import ridge_regression_grid

# fama_functions.py
from utils import ridge_regression_fast, standardize_columns
```

## File I/O Summary

### Files Read

**Panel Data (via `factor_utils.load_panel_data()`):**
- `{DATA_DIR}/arrays_{panel_id}.pkl` - Primary panel data source
  - Contains: panel DataFrame, arrays dict with model outputs
  - Format: `{'panel': pd.DataFrame, 'arrays': dict, ...}`

- `{DATA_DIR}/panel_{panel_id}.pkl` - Legacy panel data format
  - Fallback if arrays file not found

**SDF Moments (via `portfolio_stats.load_precomputed_moments()`):**
- `{DATA_DIR}/moments_{panel_id}.pkl` - Pre-computed conditional moments
  - Contains: `{month: {'rp', 'cond_var', 'second_moment', 'second_moment_inv', ...}}`

### Files Written

**Factor Results (via `factor_utils.save_factor_results()`):**
- Output location specified by caller (typically `{DATA_DIR}/{panel_id}_{method}.pkl`)
- Common outputs:
  - `{DATA_DIR}/{panel_id}_fama.pkl` - Fama-French & Fama-MacBeth results
  - `{DATA_DIR}/{panel_id}_dkkm_{nfeatures}.pkl` - DKKM results

## Performance Notes

### Parallel Processing
- All factor computation functions use `joblib.Parallel` for multi-core processing
- Configure via `n_jobs` parameter (typically 10 cores)

### Numba Acceleration
- DKKM uses Numba-accelerated RFF computation when available (2-3x speedup)
- Automatically falls back to standard numpy if Numba unavailable

### Ridge Regression
- Uses optimized ridge regression from `utils/`
- Automatically selects randomized SVD for large feature spaces (D > 1000)

## Key Design Patterns

1. **Separation of Concerns**: Each module handles a specific methodology (Fama vs DKKM)

2. **Shared Infrastructure**: `factor_utils.py` provides common utilities to avoid duplication

3. **Consistent Interface**: All factor computation functions have similar signatures

4. **Dependency Injection**: Portfolio statistics functions accept pre-computed factors

5. **Configuration-Driven**: Uses `config.ModelConfig` for model-specific parameters
