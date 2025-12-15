# KP14 Model Utilities

This directory contains all KP14 (Kogan-Papanikolaou 2014) model-specific functions.

## Files

- **panel_functions_kp14.py** - Panel data generation for KP14 model
- **sdf_compute_kp14.py** - Stochastic discount factor (SDF) computation
- **loadings_compute_kp14.py** - Factor loadings computation

## Usage

Import from parent directory:

```python
# In generate_panel.py or calculate_moments.py
from utils_kp14 import panel_functions_kp14 as panel_module
from utils_kp14 import sdf_compute_kp14 as sdf_module
```

Within this subdirectory, use relative imports:

```python
# In panel_functions_kp14.py
from .sdf_compute_kp14 import *
from .loadings_compute_kp14 import *
```

## Model Characteristics

The KP14 model focuses on innovation and technology-based factors.

See `config.py` for full KP14 configuration.
