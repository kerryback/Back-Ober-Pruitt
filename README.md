
## Usage

The workflow for each model/panel consists of five steps:

1. Generate panel data: `generate_panel.py`
2. Calculate moments: `calculate_moments.py`
3. Compute Fama-French and Fama-MacBeth factors: `run_fama.py`
4. Compute DKKM factors (for all configured feature counts): `run_dkkm.py`
5. Compute IPCA factors (for all configured K values): `run_ipca.py`

Each step generates a dictionary saved as a pickle file in the `outputs` directory.  The pickle files are read in subsequent steps. The outputs do not contain a panel index column.  Instead, the panel index is expected to be saved as a part of filenames.  Separate pickle files are generated for each panel with names `{model}_{index}_panel.pkl`, `{model}_{index}_moments.pkl`, `{model}_{index}_fama.pkl`, etc.

The `generate_panel.py` and `calculate_moments.py` are orchestration files for specialized `generate_panel_{model}.py` and `calculate_moments_{model}.py` scripts, where model is `bgn` or `kp14` or `gs21`.  There are separate `utils` directories containing these and other specialized scripts for the three models.  There is also a `utils_factors` directory containing scripts used in `run_fama`, `run_dkkm`, and `run_ipca`.

The entire workflow can be run for one or more panels of a model using 

```
python main.py {model} {start_index} {end_index}
```

where the usual python convention of counting up to but not including `end_index` is followed.  For example,

```
python main.py kp14 0 5
```
creates five panels for the kp14 model indexed as 0, 1, 2, 3, 4.  `main.py` logs progress to `logs\{model}_{start_index}_{end_index}.log.  After a panel is completed, it deletes the moments.pkl file to save disk space.


Rather than running `main.py`, the five steps can be executed individually:

1.  `python generate_panel.py {model} {n}` generates a panel with index n for the specified model.
2.  `python calculate_moments.py {model}_{n} reads  {model}_{n}_panel.pkl and calculates moments.
3.  `python run_fama {model_n}` reads {model}_{n}_panel.pkl and {model}_{n}_moments.pkl and generates Fama-French and Fama-MacBeth factors.
4.  `python run_dkkm {model_n} {k}` reads {model}_{n}_panel.pkl and {model}_{n}_moments.pkl and generates k DKKM factors.
5. `python run_ipca {model_n} {k}` reads {model}_{n}_panel.pkl and {model}_{n}_moments.pkl and generates k IPCA factors.

## Implementation Details

1. IPCA is implemented using pymanopt rather than sequential least squares.  There is no parallelization.  The solution for each group of 360 months is used to initialize pymanopt for the next group.
2. Randomized SVD is used in ridge regression when the DKKM factors exceeds a threshold (controlled by config.py parameters)
3. Some numba acceleration is used in `run_dkkm.py`
4. `calculate_moments.py` processes months in chunks (controlled by config.py parameter) and writes each chunk to disk before proceeding to the next chunk. This is to conserve RAM.  Within each chunk, months are processed in parallel (controlled by n_jobs in config.py).  At the end of the script, the chunks are read and compiled into a single pickle file.  The intermediate pickle files are deleted.
5.  If book value at $t-1$ is zero, then ROE and asset growth at $t$ are set to zero.  The number of firm/months with zero book value is computed for each panel and logged.  The default value of BURNIN was raised from 200 to 300 to reduce the frequency of zero book values.
6.  Parameters in config.py determine the type of factors computed (rank-standardized or non-rank-standardized, including market or not including market),  The defaults are to rank-standardize and to include the market.  This differs from the original code that calculated both rank-standardized and non-rank-standardized factors.

## Configuration

All parameters are centralized in [`config.py`](config.py):

### Panel Dimensions

```python
N = 1000        # Number of firms
T = 720         # Time periods (months, excluding burnin)
BGN_BURNIN = 300   # Burnin period for BGN
KP14_BURNIN = 300  # Burnin period for KP14
GS21_BURNIN = 300  # Burnin period for GS21
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

### Ridge Parameters

```python
# Shrinkage penalties (Berk-Jameson regularization)
ALPHA_LST = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]  # For BGN/KP14
ALPHA_LST_GS = [0, 0.0000001, ..., 0.1, 1]          # For GS21
IPCA_ALPHA_LST = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]
RIDGE_SVD_THRESHOLD = 1000  # Use randomized SVD when # factors > threshold
RIDGE_SVD_RANK = 500  # Rank approximation for randomized SVD (uses this many singular values)
```

### Model Parameters

Each model has calibrated parameters (lines 85-159 in `config.py`).

## Testing

The `tests/` directory contains test infrastructure and comparison utilities.

### Code Organization Comparison

**Refactored code** (this directory):
- Modular workflow: separate scripts for each step
- Organized utilities: `utils_bgn/`, `utils_kp14/`, `utils_gs21/`, `utils_factors/`
- Data in `data/` subdirectory

**Original code** (parent directory `../`):
- Monolithic `main.py` orchestrating entire workflow
- Flat structure: all function files in root directory
- Data files (Jstar.csv, G_func.csv, etc.) in root

### Test Utilities

The `tests/test_utils/` directory provides comparison tools for numerical validation:

**comparison.py** - Functions for comparing outputs:
```python
assert_close(arr1, arr2, rtol=1e-10, atol=1e-12, name="")
assert_dataframes_equal(df1, df2, rtol=1e-10, atol=1e-12)
assert_factors_equal_up_to_sign(f1, f2, rtol=1e-8, atol=1e-10)  # For IPCA
compute_summary_stats(data_dict, name="")
print_comparison_summary(stats1, stats2)
```

**config_override.py** - Small test parameters:
- N=50 firms (vs 1000 production)
- T=400 periods (vs 720 production)
- Burnin=100 (vs 300 production)
- Fixed seed=12345 for reproducibility

### Recommended Numerical Tolerances

Different operations require different precision:
- Panel generation: rtol=1e-14 (exact match expected)
- Moments: rtol=1e-10 (matrix inversions accumulate error)
- Fama/DKKM: rtol=1e-10 (matrix operations)
- IPCA: rtol=1e-6 (optimization-based, more tolerance needed)

**Note**: IPCA factors are identified only up to sign (±1). Use `assert_factors_equal_up_to_sign()` for comparison.

### Other Tests

**test_randomized_ridge.py** - Validates randomized SVD ridge regression:
- Tests basic functionality (shape, no NaN/Inf, shrinkage effect)
- Verifies consistency with fixed random seed
- Benchmarks performance for production-scale problems

Run with:
```bash
cd tests
python test_randomized_ridge.py
```
