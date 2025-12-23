# Regression Testing Framework

This document describes the regression testing framework for validating that the refactored code produces identical results to the original code.

## Overview

The regression tests compare outputs from the refactored code (current directory) against the original code (parent directory `../`) for all three models (BGN, KP14, GS21) across all 5 workflow steps.

**Key Features:**
- Deterministic execution with fixed random seeds
- Small test cases for fast execution
- Appropriate numerical tolerances for each workflow step
- Comprehensive coverage of all models and steps
- Informative error messages with detailed diagnostics

## Test Structure

### Test Files

```
tests/
├── test_utils/
│   ├── __init__.py           # Package initialization
│   ├── comparison.py         # Comparison utilities
│   └── config_override.py    # Test configuration
├── test_regression_bgn.py    # BGN model regression tests
├── test_regression_kp14.py   # KP14 model regression tests
├── test_regression_gs21.py   # GS21 model regression tests
├── test_randomized_ridge.py  # Randomized SVD validation
└── run_all_regression_tests.py  # Master test runner
```

### Test Utilities

**comparison.py** - Comparison functions with informative error messages:
- `assert_close()` - Compare numpy arrays with tolerance
- `assert_dataframes_equal()` - Compare pandas DataFrames column-by-column
- `assert_factors_equal_up_to_sign()` - Compare IPCA factors allowing sign flips
- `compute_summary_stats()` - Compute summary statistics for data dictionary
- `print_comparison_summary()` - Print comparison summary

**config_override.py** - Small parameter values for fast testing:
```python
TEST_N = 50              # Number of firms (vs 1000 production)
TEST_T = 200             # Time periods (vs 720 production)
TEST_BURNIN = 100        # Burnin period (vs 300 production)
TEST_SEED = 12345        # Fixed random seed
TEST_PANEL_ID = 999      # Test panel ID
TEST_DKKM_FEATURES = [6, 36]     # Subset for speed
TEST_IPCA_K_VALUES = [1, 2]      # Subset for speed
TEST_N_JOBS = 2          # Parallel jobs (vs 10 production)
```

## Test Execution

### Running All Tests

```bash
cd tests
python run_all_regression_tests.py
```

This runs all three model regression tests and provides a comprehensive summary.

### Running Specific Models

```bash
# Run only BGN tests
python run_all_regression_tests.py bgn

# Run KP14 and GS21 tests
python run_all_regression_tests.py kp14 gs21
```

### Running Individual Test Suites

```bash
# Run BGN regression tests
python test_regression_bgn.py

# Run KP14 regression tests
python test_regression_kp14.py

# Run GS21 regression tests
python test_regression_gs21.py
```

## Test Coverage

Each model regression test validates all 5 workflow steps:

### 1. Panel Generation (test_01_panel_generation)
- Runs: `generate_panel.py` (refactored) vs `generate_panel_{model}.py` (original)
- Compares: Panel DataFrame and all arrays (A_taylor, A_proj, f_taylor, f_proj)
- Tolerance: rtol=1e-14, atol=1e-15 (exact match expected)

### 2. Moment Calculation (test_02_moments)
- Runs: `calculate_moments.py` (refactored) vs `calculate_moments_{model}.py` (original)
- Compares: rp, cond_var, second_moment, second_moment_inv
- Tolerance: rtol=1e-10 for matrix inversions, rtol=1e-12 otherwise

### 3. Fama Factors (test_03_fama)
- Runs: `run_fama.py` on both versions
- Compares: All factor arrays in output
- Tolerance: rtol=1e-10, atol=1e-12

### 4. DKKM Factors (test_04_dkkm)
- Runs: `run_dkkm.py` for each n_features in TEST_DKKM_FEATURES
- Compares: All factor arrays for each feature count
- Tolerance: rtol=1e-10, atol=1e-12

### 5. IPCA Factors (test_05_ipca)
- Runs: `run_ipca.py` for each K in TEST_IPCA_K_VALUES
- Compares: All arrays, with sign-invariant comparison for factors
- Tolerance: rtol=1e-6, atol=1e-8 (optimization-based method)

## Numerical Tolerances

Different workflow steps require different tolerances:

| Step | Relative Tol | Absolute Tol | Reason |
|------|-------------|--------------|--------|
| Panel Generation | 1e-14 | 1e-15 | Direct calculations, exact match expected |
| Moments | 1e-10 | 1e-9 | Matrix inversions accumulate error |
| Fama Factors | 1e-10 | 1e-12 | Matrix operations |
| DKKM Factors | 1e-10 | 1e-12 | Ridge regression with randomized SVD |
| IPCA Factors | 1e-6 | 1e-8 | Stiefel manifold optimization |

## Special Considerations

### IPCA Sign Indeterminacy

IPCA factors are identified only up to sign (±1). The comparison function `assert_factors_equal_up_to_sign()` checks if factors match either as-is or with sign flipped:

```python
# Check if equal or negated
positive_match = np.allclose(f1, f2, rtol=rtol, atol=atol)
negative_match = np.allclose(f1, -f2, rtol=rtol, atol=atol)

if not (positive_match or negative_match):
    raise AssertionError(...)
```

### Random Seed Handling

All tests use `TEST_SEED = 12345` for reproducibility. This ensures:
- Panel generation produces identical data
- Random Fourier Features (DKKM) are deterministic
- IPCA random restarts are reproducible

### Parallel Processing

Tests use `TEST_N_JOBS = 2` instead of production value (10) to:
- Reduce overhead for small test cases
- Ensure deterministic ordering
- Make tests faster

## Expected Test Duration

With test configuration (N=50, T=200, BURNIN=100):

| Test | Approximate Duration |
|------|---------------------|
| Panel Generation | ~5 seconds |
| Moment Calculation | ~10 seconds |
| Fama Factors | ~2 seconds |
| DKKM Factors (2 configs) | ~15 seconds |
| IPCA Factors (2 K values) | ~30 seconds |
| **Total per model** | **~60 seconds** |
| **All 3 models** | **~3 minutes** |

## Interpreting Test Results

### Successful Test Output

```
======================================================================
TEST 1: Panel Generation
======================================================================

[1/2] Running refactored code...
[OK] Refactored code completed

[2/2] Running original code...
[OK] Original code completed

[COMPARE] Loading results...
  Refactored keys: ['panel', 'A_1_taylor', 'A_2_taylor', ...]
  Original keys: ['panel', 'A_1_taylor', 'A_2_taylor', ...]

[COMPARE] Comparing panel DataFrame...
  [PASS] Panel DataFrames match

[COMPARE] Comparing arrays...
  [PASS] A_1_taylor matches
  [PASS] A_2_taylor matches
  ...

======================================================================
[PASS] Panel generation test passed
======================================================================
```

### Failed Test Output

When a test fails, you'll see detailed diagnostics:

```
AssertionError: Column 'ret' mismatch:
  Max absolute diff: 1.23e-09
  Max relative diff: 4.56e-08
  Tolerance: rtol=1e-14, atol=1e-15
```

This indicates:
- Which array/column failed
- Maximum absolute and relative differences
- Expected tolerances

## Troubleshooting

### Common Issues

1. **Test fails with "file not found"**
   - Ensure original code exists in parent directory (`../`)
   - Verify file names match expected pattern

2. **Numerical tolerance exceeded**
   - Check if differences are small and acceptable
   - May need to adjust tolerance for that specific step
   - Investigate if implementation differs in meaningful way

3. **IPCA sign mismatch**
   - Normal for factors to flip sign
   - Test automatically handles ±1 correlation
   - If test fails, factors may differ structurally

4. **Random seed not working**
   - Verify TEST_SEED is used in both versions
   - Check that RNG state is identical at comparison point
   - Parallel processing must be deterministic

### Debugging Tests

To debug a failing test:

1. Run the specific test file directly:
   ```bash
   python test_regression_bgn.py
   ```

2. Check the pickle files manually:
   ```bash
   python ../view_pickle.py outputs/bgn_999_panel.pkl
   ```

3. Add print statements in comparison functions to see values

4. Use Python debugger:
   ```bash
   python -m pdb test_regression_bgn.py
   ```

## Extending Tests

### Adding a New Model

1. Copy an existing test file (e.g., `test_regression_bgn.py`)
2. Update class name and model references
3. Update array keys to match new model's output
4. Add to `run_all_regression_tests.py`

### Adding a New Workflow Step

1. Add new test method to each model test class
2. Follow naming convention: `test_0X_stepname()`
3. Update `tests` list in `main()` function
4. Document in this file and README.md

### Modifying Tolerances

If numerical tolerances need adjustment:

1. Update tolerance in specific test method
2. Add comment explaining why tolerance changed
3. Document in this file under "Numerical Tolerances"

## Maintenance

### When to Run Tests

- Before committing significant changes
- After refactoring any workflow step
- Before creating a release
- When debugging numerical discrepancies

### Updating Test Configuration

If production parameters change, update `config_override.py` to maintain:
- N/T ratio similar to production
- Burnin fraction similar to production
- Feature counts representative of production

### Version Control

All test files should be committed to git:
```bash
git add tests/
git commit -m "Add comprehensive regression test suite"
```

Do NOT commit test output files (`outputs/` directory).
