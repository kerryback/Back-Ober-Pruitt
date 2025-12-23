"""
Test configuration overrides.

Provides small parameter values for fast regression testing.
"""

# Small panel dimensions for fast testing
TEST_N = 50          # Number of firms (vs 1000 in production)
TEST_T = 400         # Time periods (vs 720 in production)
TEST_BURNIN = 100    # Burnin period (vs 300 in production)

# Fixed seed for reproducibility
TEST_SEED = 12345

# Test panel identifier
TEST_PANEL_ID = 999  # Use 999 to avoid conflicts with production panels

# DKKM test configurations (subset for speed)
TEST_DKKM_FEATURES = [6, 36]  # vs [6, 36, 360] in production

# IPCA test configurations (subset for speed)
TEST_IPCA_K_VALUES = [1, 2]   # vs [1, 2, 3] in production

# Number of parallel jobs for testing
TEST_N_JOBS = 2  # vs 10 in production
