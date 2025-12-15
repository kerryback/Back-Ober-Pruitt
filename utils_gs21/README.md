# GS21 Model Utilities

This directory contains all GS21 (Gomes-Schmid 2021) model-specific functions.

## Files

- **panel_functions_gs21.py** - Panel data generation for GS21 model
- **sdf_compute_gs21.py** - Stochastic discount factor (SDF) computation
- **loadings_compute_gs21.py** - Factor loadings computation

## Usage

Import from parent directory:

```python
# In generate_panel.py or calculate_moments.py
from utils_gs21 import panel_functions_gs21 as panel_module
from utils_gs21 import sdf_compute_gs21 as sdf_module
```

Within this subdirectory, use relative imports:

```python
# In panel_functions_gs21.py
from .sdf_compute_gs21 import *
from .loadings_compute_gs21 import *
```

## Model Characteristics

The GS21 model focuses on investment-based and corporate finance features.

See `config.py` for full GS21 configuration.
