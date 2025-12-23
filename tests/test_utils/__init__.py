"""Test utilities for regression testing."""

from .comparison import (
    assert_close,
    assert_dataframes_equal,
    assert_factors_equal_up_to_sign,
    compute_summary_stats,
    print_comparison_summary
)

from .config_override import (
    TEST_N,
    TEST_T,
    TEST_BURNIN,
    TEST_SEED,
    TEST_PANEL_ID,
    TEST_DKKM_FEATURES,
    TEST_IPCA_K_VALUES,
    TEST_N_JOBS
)

__all__ = [
    'assert_close',
    'assert_dataframes_equal',
    'assert_factors_equal_up_to_sign',
    'compute_summary_stats',
    'print_comparison_summary',
    'TEST_N',
    'TEST_T',
    'TEST_BURNIN',
    'TEST_SEED',
    'TEST_PANEL_ID',
    'TEST_DKKM_FEATURES',
    'TEST_IPCA_K_VALUES',
    'TEST_N_JOBS',
]
