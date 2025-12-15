# BGN Model Utilities

This directory contains all BGN (Baseline Growth and Noise) model-specific functions.

## Files

- **panel_functions_bgn.py** - Panel data generation for BGN model
- **sdf_compute_bgn.py** - Stochastic discount factor (SDF) computation
- **loadings_compute_bgn.py** - Factor loadings computation
- **vasicek.py** - Vasicek interest rate model (used by BGN)

## Usage

Import from parent directory:

```python
# In generate_panel.py or calculate_moments.py
from utils_bgn import panel_functions_bgn as panel_module
from utils_bgn import sdf_compute_bgn as sdf_module
```

Within this subdirectory, use relative imports:

```python
# In panel_functions_bgn.py
from .vasicek import *
from .sdf_compute_bgn import *
from .loadings_compute_bgn import *
```

## Model Characteristics

The BGN model includes 5 characteristics:
- Investment/assets ratio
- Book-to-market ratio
- Profitability
- Size
- Model-specific factors

See `config.py` for full BGN configuration.
