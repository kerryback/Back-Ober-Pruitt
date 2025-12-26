# Regression Testing Framework

This document describes the regression testing framework for validating that the refactored code produces identical results to the original code.

## Overview

The regression tests compare outputs from the refactored code (`utils_bgn/`, `utils_kp14/`, `utils_gs21/`) against the original code (`tests/original_code/`) for all three models (BGN, KP14, GS21).

**Key Features:**
- Deterministic execution with fixed random seeds
- Small test cases for fast execution (N=50, T=400)
- Appropriate numerical tolerances for each workflow step
- Comprehensive coverage of all models
- Informative error messages with detailed diagnostics

## Test Structure

### Test Files

```
tests/
├── test_utils/
│   ├── __init__.py           # Package initialization
│   └── comparison.py         # Comparison utilities
├── original_code/            # Original code to compare against
│   ├── panel_functions.py    # BGN panel generation (original)
│   ├── panel_functions_kp14.py
│   ├── panel_functions_gs21.py
│   ├── sdf_compute.py        # BGN SDF calculations (original)
│   ├── sdf_compute_kp14.py
│   ├── sdf_compute_gs21.py
│   ├── Jstar.csv             # BGN data file
│   ├── G_func.csv            # KP14 data file
│   ├── integ_results.npz     # KP14 data file
│   └── GS21_solfiles/        # GS21 data files
├── test_panel_bgn.py         # BGN panel generation test
├── test_panel_kp14.py        # KP14 panel generation test
├── test_panel_gs21.py        # GS21 panel generation test
├── test_moments_bgn.py       # BGN moment calculation test
├── test_moments_kp14.py      # KP14 moment calculation test
├── test_moments_gs21.py      # GS21 moment calculation test
├── test_fama.py              # Fama factor regression test
├── test_randomized_ridge.py  # Randomized SVD validation
├── run_generate_panel.py     # Helper script for panel generation
└── TESTING.md                # This file
```

### Test Utilities

**comparison.py** - Comparison functions with informative error messages:
- `assert_close()` - Compare numpy arrays with tolerance
- `assert_dataframes_equal()` - Compare pandas DataFrames column-by-column
- `assert_factors_equal_up_to_sign()` - Compare IPCA factors allowing sign flips
- `compute_summary_stats()` - Compute summary statistics for data dictionary
- `print_comparison_summary()` - Print comparison summary

## Test Configuration

All tests use configuration values from `config.py`:
```python
N = 50                   # Number of firms (configured in config.py)
T = 400                  # Time periods (configured in config.py)
BGN_BURNIN = 300        # Burnin period for BGN
KP14_BURNIN = 300       # Burnin period for KP14
GS21_BURNIN = 300       # Burnin period for GS21
TEST_SEED = 12345       # Fixed random seed (set in each test script)
```

## Test Execution

### Panel Generation Tests

Test that refactored panel generation produces identical results to original code.

```bash
cd tests

# Run individual panel tests
python test_panel_bgn.py
python test_panel_kp14.py
python test_panel_gs21.py
```

**What they test:**
- Generate panels with both current and original code using same random seed
- Compare all panel DataFrame columns
- Compare all arrays (A_taylor, A_proj, f_taylor, f_proj, K matrix for KP14, etc.)
- Tolerance: rtol=1e-14, atol=1e-15 (exact match expected)

**Note:** BGN, KP14, and GS21 tests may report NaN handling improvements where refactored code sets `roe=0` and `agr=0` when book value is 0, instead of NaN. This is an expected improvement.

### Moment Calculation Tests

Test that refactored SDF moment calculations produce identical results to original code.

```bash
cd tests

# Run individual moment tests
python test_moments_bgn.py
python test_moments_kp14.py
python test_moments_gs21.py
```

**What they test:**
1. Generate panel via subprocess: `run_generate_panel.py <model> <panel_id>`
2. Calculate moments via subprocess: `calculate_moments.py <model>_<panel_id>`
3. Read pickled moments (current code output)
4. Compute moments with original code (in-process import)
5. Compare rp (risk premia), cond_var (conditional variance), max_sr (maximum Sharpe ratio)
6. Cleanup: Delete panel and moments pickle files
- Tolerance: rtol=1e-12, atol=1e-14

**Execution pattern:**
- **Current code**: Runs via subprocess from terminal (`calculate_moments.py`)
- **Original code**: Imported directly and called in-process

### Fama Factor Tests

Test Fama-French and Fama-MacBeth factor calculations.

```bash
cd tests

# Run Fama tests for each model
python test_fama.py bgn
python test_fama.py kp14
python test_fama.py gs21
```

**What they test:**
1. Generate panel via subprocess
2. Compute Fama factors with original code (in-process)
3. Compute Fama factors with current code via subprocess (`run_fama.py`)
4. Compare factor returns
5. Cleanup: Delete pickle files
- Tolerance: rtol=1e-14, atol=1e-15

## Numerical Tolerances

Different workflow steps require different tolerances:

| Step | Relative Tol | Absolute Tol | Reason |
|------|-------------|--------------|--------|
| Panel Generation | 1e-14 | 1e-15 | Direct calculations, exact match expected |
| Moments | 1e-12 | 1e-14 | Matrix operations accumulate small errors |
| Fama Factors | 1e-14 | 1e-15 | Direct calculations |

## Expected Test Duration

With test configuration (N=50, T=400, BURNIN=300):

| Test | Approximate Duration |
|------|---------------------|
| Panel Generation (each model) | ~5 seconds |
| Moment Calculation (each model) | ~30 seconds |
| Fama Factors (each model) | ~5 seconds |
| **Total per model** | **~40 seconds** |
| **All 3 models** | **~2 minutes** |

## Interpreting Test Results

### Successful Test Output

```
======================================================================
BGN Panel Generation
======================================================================
N=50, T=400, Burnin=300, Seed=12345

[1/4] Generating panel with current code...
      Panel shape: (20000, 12)

[2/4] Generating panel with original code...
      Panel shape: (20000, 12)

[3/4] Comparing panels for numerical identity...
  [PASS] panel[month]
  [PASS] panel[firmid]
  [PASS] panel[mve]
  ...
  [PASS] arrays[0]
  [PASS] arrays[1]
  ...

[4/4] Saving panel and arrays to outputs/bgn_test_panel.pkl...
      Saved successfully

======================================================================
[DONE] BGN panel generation complete
======================================================================
```

### Failed Test Output

When a test fails, you'll see detailed diagnostics:

```
  [FAIL] panel[xret]: Max diff=1.23e-09, rtol=1.23e-08, atol=1e-15
         Expected: rtol=1e-14, atol=1e-15
```

This indicates:
- Which array/column failed
- Maximum difference and relative/absolute tolerances
- Expected tolerances

### NaN Handling Notes

Some tests report improved NaN handling:

```
  [FAIL] panel[roe]: NaN locations differ (expected improvement)
         New code: 0 NaNs, Original code: 217 NaNs
```

This is **expected** - the refactored code sets `roe=0` and `agr=0` when book value is 0, instead of propagating NaN. The test will still show "[NOTE] Arrays are identical; panel differences are only NaN handling (expected improvement)".

## Workflow Comparison

### Original Code (tests/original_code/)
- **Implementation**: Import directly and call functions in-process
- **Data files**: Uses relative paths (Jstar.csv, G_func.csv, etc.)
- **Execution**: Must `os.chdir()` to original_code/ before importing

### Current Code (utils_bgn/, utils_kp14/, utils_gs21/)
- **Implementation**: Run from terminal via subprocess
- **Data files**: Uses absolute paths via config.DATA_DIR
- **Execution**: Call via subprocess (`generate_panel.py`, `calculate_moments.py`, etc.)

## Directory Navigation Pattern

Tests that import original code use this pattern:

```python
# Change to original_code directory so data files can be found
original_cwd = os.getcwd()
original_code_dir = Path(__file__).parent / 'original_code'
os.chdir(original_code_dir)

# Import original code
import panel_functions as panel_old

# Change back
os.chdir(original_cwd)
```

This is necessary because original code uses relative paths like `pd.read_csv('Jstar.csv')` which look in the current working directory.

## Troubleshooting

### Common Issues

1. **Test fails with "file not found"**
   - Ensure original code exists in `tests/original_code/`
   - Verify data files are in `tests/original_code/` (Jstar.csv, G_func.csv, etc.)

2. **Numerical tolerance exceeded**
   - Check if differences are small and acceptable
   - May need to adjust tolerance for that specific step
   - Investigate if implementation differs in meaningful way

3. **Random seed not working**
   - Verify TEST_SEED is used in both versions
   - Check that RNG state is identical at comparison point

4. **Subprocess failures**
   - Check stdout/stderr from subprocess.run()
   - Verify scripts exist: `run_generate_panel.py`, `calculate_moments.py`
   - Ensure pickle files are being created/deleted properly

### Debugging Tests

To debug a failing test:

1. Run the specific test file directly:
   ```bash
   cd tests
   python test_panel_bgn.py
   ```

2. Check the pickle files manually:
   ```bash
   python
   >>> import pickle
   >>> with open('outputs/bgn_test_panel.pkl', 'rb') as f:
   ...     data = pickle.load(f)
   >>> data.keys()
   ```

3. Add print statements in comparison functions to see values

4. Use Python debugger:
   ```bash
   python -m pdb test_panel_bgn.py
   ```

## Maintenance

### When to Run Tests

- Before committing significant changes
- After refactoring any workflow step
- Before creating a release
- When debugging numerical discrepancies

### Version Control

All test files should be committed to git:
```bash
git add tests/
git commit -m "Update regression test suite"
```

Do NOT commit test output files (`outputs/` directory) or pickle files.
