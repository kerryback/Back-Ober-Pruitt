"""
Run main.py for all three models (BGN, KP14, GS21) and log output.

This script runs the complete workflow for each model sequentially and
captures all output to outputs/output.txt.

Usage:
    python run_all_models.py
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def run_model(model_name, output_file):
    """Run main.py for a specific model and log output."""
    print(f"\n{'='*70}")
    print(f"Starting {model_name.upper()} model")
    print(f"{'='*70}")

    header = f"\n{'='*70}\n"
    header += f"{model_name.upper()} MODEL - Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}\n"
    header += f"{'='*70}\n\n"

    # Write header to output file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(header)

    # Run main.py for this model
    cmd = [sys.executable, 'main.py', model_name]
    print(f"Running: {' '.join(cmd)}")

    start_time = time.time()

    # Run and capture output
    with open(output_file, 'a', encoding='utf-8') as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

    elapsed = time.time() - start_time

    # Write footer to output file
    footer = f"\n{'='*70}\n"
    footer += f"{model_name.upper()} MODEL - Completed at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}\n"
    footer += f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)\n"
    footer += f"Return code: {result.returncode}\n"
    footer += f"{'='*70}\n\n"

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(footer)

    print(f"[OK] {model_name.upper()} completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    if result.returncode != 0:
        print(f"[ERROR] {model_name.upper()} failed with return code {result.returncode}")
        return False

    return True

def main():
    """Main execution function."""
    overall_start = time.time()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    output_file = os.path.join(outputs_dir, 'output.txt')

    # Write initial header
    initial_header = f"{'='*70}\n"
    initial_header += f"RUNNING ALL MODELS (BGN, KP14, GS21)\n"
    initial_header += f"{'='*70}\n"
    initial_header += f"Started at: {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}\n"
    initial_header += f"Output file: {output_file}\n"
    initial_header += f"{'='*70}\n\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(initial_header)

    print(initial_header)

    # Run each model
    models = ['bgn', 'kp14', 'gs21']
    results = {}

    for model in models:
        success = run_model(model, output_file)
        results[model] = success

        if not success:
            print(f"\n[ERROR] Stopping execution due to {model.upper()} failure")
            break

    # Final summary
    overall_elapsed = time.time() - overall_start

    final_summary = f"\n{'='*70}\n"
    final_summary += f"ALL MODELS COMPLETE\n"
    final_summary += f"{'='*70}\n"
    final_summary += f"Finished at: {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}\n"
    final_summary += f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f}min)\n"
    final_summary += f"\nResults:\n"
    for model, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        final_summary += f"  {model.upper()}: {status}\n"
    final_summary += f"{'='*70}\n"

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(final_summary)

    print(final_summary)
    print(f"\nAll output written to: {output_file}")

if __name__ == "__main__":
    main()
