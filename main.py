"""
NoIPCA - Master Orchestration Script

Runs the complete workflow for a range of panel identifiers:
1. Generate panel data
2. Calculate SDF moments
3. Compute Fama factors
4. Compute DKKM factors for multiple feature counts

Usage:
    python main.py [model] [start] [end]

Arguments:
    model: Model name (bgn, kp14, gs21) - case insensitive
    start: Starting index (optional, default: 0)
    end: Ending index (optional, default: 1)

    Runs workflow for panel_id in range(start, end)

Examples:
    python main.py bgn           # Runs for index 0 only
    python main.py bgn 0 5       # Runs for indices 0, 1, 2, 3, 4
    python main.py kp14 10 15    # Runs for indices 10, 11, 12, 13, 14

Output:
    All output is logged to: logs/log_{model}_{start}_{end}.txt
"""

import sys
import os
import subprocess
import time
from datetime import datetime
from config import DATA_DIR, N_DKKM_FEATURES_LIST


def run_script(script_name, args, description):
    """Run a Python script and handle errors. Returns elapsed time in seconds."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")

    cmd = [sys.executable, script_name] + args
    print(f"Running: {' '.join(cmd)}")
    print(f"Started at {datetime.now().strftime('%I:%M%p')}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"[OK] Completed at {datetime.now().strftime('%I:%M%p')} (took {elapsed:.1f}s / {elapsed/60:.1f}min)")
    return elapsed


def run_workflow_for_index(model, panel_id):
    """Run the complete workflow for a single panel index."""
    full_panel_id = f"{model}_{panel_id}"

    print(f"\n{'='*70}")
    print(f"RUNNING WORKFLOW FOR {full_panel_id.upper()}")
    print(f"{'='*70}")

    # Track timing for each step
    timings = {}

    # Step 1: Generate panel data
    timings['generate_panel'] = run_script(
        "generate_panel.py",
        [model, str(panel_id)],
        f"STEP 1: Generating {model.upper()} panel data (index={panel_id})"
    )

    # Step 2: Calculate SDF conditional moments
    timings['calculate_moments'] = run_script(
        "calculate_moments.py",
        [full_panel_id],
        "STEP 2: Calculating SDF conditional moments (rp, cond_var, etc.)"
    )

    # Step 3: Compute Fama factors
    timings['run_fama'] = run_script(
        "run_fama.py",
        [full_panel_id],
        "STEP 3: Computing Fama-French and Fama-MacBeth factors"
    )

    # Step 4: Compute DKKM factors for each feature count
    timings['run_dkkm'] = {}
    for i, nfeatures in enumerate(N_DKKM_FEATURES_LIST, 1):
        timings['run_dkkm'][nfeatures] = run_script(
            "run_dkkm.py",
            [full_panel_id, str(nfeatures)],
            f"STEP 4.{i}: Computing DKKM factors (nfeatures={nfeatures})"
        )

    # Print summary for this index
    total_time = sum([timings['generate_panel'], timings['calculate_moments'], timings['run_fama']] +
                     list(timings['run_dkkm'].values()))

    print(f"\n{'='*70}")
    print(f"WORKFLOW COMPLETE FOR {full_panel_id.upper()}")
    print(f"{'='*70}")
    print(f"Execution Times:")
    print(f"  1. Generate panel:      {timings['generate_panel']:7.1f}s ({timings['generate_panel']/60:5.1f}min)")
    print(f"  2. Calculate moments:   {timings['calculate_moments']:7.1f}s ({timings['calculate_moments']/60:5.1f}min)")
    print(f"  3. Fama factors:        {timings['run_fama']:7.1f}s ({timings['run_fama']/60:5.1f}min)")
    for i, nfeatures in enumerate(N_DKKM_FEATURES_LIST, 1):
        dkkm_time = timings['run_dkkm'][nfeatures]
        print(f"  4.{i} DKKM (n={nfeatures:4d}):      {dkkm_time:7.1f}s ({dkkm_time/60:5.1f}min)")
    print(f"  {'-'*50}")
    print(f"  Total for {full_panel_id}:  {total_time:7.1f}s ({total_time/60:5.1f}min)")
    print(f"{'='*70}\n")

    return timings


def main():
    """Main orchestration function."""
    overall_start = time.time()

    # Parse arguments
    if len(sys.argv) < 2:
        print("ERROR: Model name required")
        print("\nUsage: python main.py [model] [start] [end]")
        print("  model: bgn, kp14, or gs21 (case insensitive)")
        print("  start: starting index (optional, default: 0)")
        print("  end: ending index (optional, default: 1)")
        print("\nExamples:")
        print("  python main.py bgn           # Runs for index 0")
        print("  python main.py bgn 0 5       # Runs for indices 0-4")
        print("  python main.py kp14 10 15    # Runs for indices 10-14")
        sys.exit(1)

    # Get model name (case insensitive)
    model = sys.argv[1].lower()
    valid_models = ['bgn', 'kp14', 'gs21']

    if model not in valid_models:
        print(f"ERROR: Invalid model '{sys.argv[1]}'")
        print(f"Valid models: {', '.join(valid_models)} (case insensitive)")
        sys.exit(1)

    # Get start and end indices
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # Validate range
    if start >= end:
        print(f"ERROR: start ({start}) must be less than end ({end})")
        sys.exit(1)

    # Setup logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, f"log_{model}_{start}_{end}.txt")

    # Redirect stdout and stderr to log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_handle = open(log_file, 'w', encoding='utf-8')
    sys.stdout = log_handle
    sys.stderr = log_handle

    try:
        print("="*70)
        print("NoIPCA - COMPLETE WORKFLOW")
        print("="*70)
        print(f"Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
        print()
        print(f"Configuration:")
        print(f"  Model: {model}")
        print(f"  Index range: {start} to {end-1} (inclusive)")
        print(f"  Total runs: {end - start}")
        print(f"  DKKM features: {N_DKKM_FEATURES_LIST}")
        print(f"  Log file: {log_file}")
        print()

        # Run workflow for each index in range
        all_timings = {}
        for i in range(start, end):
            all_timings[i] = run_workflow_for_index(model, i)

        # Final summary
        overall_elapsed = time.time() - overall_start

        print(f"\n{'='*70}")
        print("ALL WORKFLOWS COMPLETE")
        print(f"{'='*70}")
        print(f"Finished at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
        print()
        print(f"Summary for {model.upper()} indices {start} to {end-1}:")
        print(f"  Total runs: {end - start}")
        print(f"  Total time: {overall_elapsed:7.1f}s ({overall_elapsed/60:5.1f}min)")
        print(f"  Average time per run: {overall_elapsed/(end-start):7.1f}s ({overall_elapsed/(end-start)/60:5.1f}min)")
        print()
        print(f"Output files created in: {DATA_DIR}")
        for i in range(start, end):
            full_panel_id = f"{model}_{i}"
            print(f"\n  Index {i} ({full_panel_id}):")
            print(f"    - Panel data: {full_panel_id}_arrays.pkl")
            print(f"    - SDF moments: {full_panel_id}_moments.pkl")
            print(f"    - Fama factors: {full_panel_id}_fama.pkl")
            for nfeatures in N_DKKM_FEATURES_LIST:
                print(f"    - DKKM (n={nfeatures}): {full_panel_id}_dkkm_{nfeatures}.pkl")
        print()
        print(f"Log file: {log_file}")
        print(f"{'='*70}")

    finally:
        # Restore stdout/stderr and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()

    # Print completion message to console
    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*70}")
    print(f"Model: {model.upper()}")
    print(f"Indices: {start} to {end-1}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)")
    print(f"Log file: {log_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
