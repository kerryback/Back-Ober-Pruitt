"""
NoIPCA - Master Orchestration Script

Runs the complete workflow:
1. Generate panel data
2. Compute Fama factors
3. Compute DKKM factors for multiple feature counts

Usage:
    python main.py [model] [panel_id]

Arguments:
    model: Model name (bgn, kp14, gs21) - case insensitive
    panel_id: Optional panel identifier (default: 0)

Examples:
    python main.py bgn
    python main.py BGN 5
    python main.py kp14 exp1
"""

import sys
import os
import subprocess
from datetime import datetime
from config import DATA_DIR, N_DKKM_FEATURES_LIST


def run_script(script_name, args, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")

    cmd = [sys.executable, script_name] + args
    print(f"Running: {' '.join(cmd)}")
    print(f"Started at {datetime.now().strftime('%I:%M%p')}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"[OK] Completed at {datetime.now().strftime('%I:%M%p')}")
    return result


def main():
    """Main orchestration function."""

    print("="*70)
    print("NoIPCA - COMPLETE WORKFLOW")
    print("="*70)
    print(f"Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print()

    # Parse arguments
    if len(sys.argv) < 2:
        print("ERROR: Model name required")
        print("\nUsage: python main.py [model] [panel_id]")
        print("  model: bgn, kp14, or gs21 (case insensitive)")
        print("  panel_id: optional identifier (default: 0)")
        print("\nExamples:")
        print("  python main.py bgn")
        print("  python main.py BGN 5")
        print("  python main.py kp14 exp1")
        sys.exit(1)

    # Get model name (case insensitive)
    model = sys.argv[1].lower()
    valid_models = ['bgn', 'kp14', 'gs21']

    if model not in valid_models:
        print(f"ERROR: Invalid model '{sys.argv[1]}'")
        print(f"Valid models: {', '.join(valid_models)} (case insensitive)")
        sys.exit(1)

    # Get panel_id (default to 0)
    panel_id = sys.argv[2] if len(sys.argv) > 2 else "0"

    # Construct full panel identifier
    full_panel_id = f"{model}_{panel_id}"

    print(f"Configuration:")
    print(f"  Model: {model}")
    print(f"  Panel ID: {panel_id}")
    print(f"  Full identifier: {full_panel_id}")
    print(f"  DKKM features: {N_DKKM_FEATURES_LIST}")
    print()

    # Change to NoIPCA directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Step 1: Generate panel data using unified orchestrator
    run_script(
        "generate_panel.py",
        [model, panel_id],
        f"STEP 1: Generating {model.upper()} panel data"
    )

    # Step 2: Calculate SDF conditional moments
    run_script(
        "calculate_moments.py",
        [full_panel_id],
        "STEP 2: Calculating SDF conditional moments (rp, cond_var, etc.)"
    )

    # Step 3: Compute Fama factors
    run_script(
        "run_fama.py",
        [full_panel_id],
        "STEP 3: Computing Fama-French and Fama-MacBeth factors"
    )

    # Step 4: Compute DKKM factors for each feature count
    for i, nfeatures in enumerate(N_DKKM_FEATURES_LIST, 1):
        run_script(
            "run_dkkm.py",
            [full_panel_id, str(nfeatures)],
            f"STEP 4.{i}: Computing DKKM factors (nfeatures={nfeatures})"
        )

    # Summary
    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print()
    print(f"Output files created in: {DATA_DIR}")
    print(f"  1. Panel data: arrays_{full_panel_id}.pkl")
    print(f"  2. SDF moments: moments_{full_panel_id}.pkl")
    print(f"  3. Fama factors: {full_panel_id}_fama.pkl")
    for i, nfeatures in enumerate(N_DKKM_FEATURES_LIST, 4):
        print(f"  {i}. DKKM (n={nfeatures}): {full_panel_id}_dkkm_{nfeatures}.pkl")
    print()
    print("To load results:")
    print("  import pickle")
    print(f"  with open('{os.path.join(DATA_DIR, full_panel_id)}_fama.pkl', 'rb') as f:")
    print("      fama_results = pickle.load(f)")
    if N_DKKM_FEATURES_LIST:
        nf = N_DKKM_FEATURES_LIST[0]
        print(f"  with open('{os.path.join(DATA_DIR, full_panel_id)}_dkkm_{nf}.pkl', 'rb') as f:")
        print("      dkkm_results = pickle.load(f)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
