"""
Wrapper to run panel generation with a fixed seed for testing.

This script calls the panel generation functions directly with TEST_SEED
to ensure reproducibility.

Usage:
    python run_generate_panel.py bgn 999
"""

import sys
import os
import numpy as np
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / 'test_utils'))

from test_utils.config_override import TEST_SEED


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_generate_panel.py <model> <identifier>")
        sys.exit(1)

    model = sys.argv[1].lower()
    identifier = int(sys.argv[2])

    # Set seed for reproducibility
    np.random.seed(TEST_SEED)

    # Import and run generate_panel.main()
    # This will use the seed we just set
    sys.argv = ['generate_panel.py', model, str(identifier)]

    # Change to parent directory so generate_panel.py runs from correct location
    os.chdir(str(Path(__file__).parent.parent))

    # Import and run
    from generate_panel import main as generate_main
    generate_main()


if __name__ == "__main__":
    main()
