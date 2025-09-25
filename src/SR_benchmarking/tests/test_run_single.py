#!/usr/bin/env python3
"""
Minimal test script for run_single function.
"""

from pathlib import Path
from run.utils import (
    ModelSettings,
    NeuralOptions,
    MutationWeights,
    DatasetSettings,
    init_pysr_model,
    run_single
)

def test_run_single_minimal():
    """Run a minimal test of the run_single function."""

    print("[INFO] Setting up minimal test configuration...")

    # 1. Create minimal model settings - very few iterations for quick test
    model_settings = ModelSettings(
        niterations=4,
        early_stopping_condition=0.0,  # Disable early stopping
        verbosity=1
    )
    print(f"[INFO] Model settings: {model_settings.niterations} iterations")

    # 2. Create neural options with neural features disabled for simplicity
    neural_options = NeuralOptions(
        active=True,
        model_path='/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-e51hcsb9.onnx',
        device='cuda',
    )
    print(f"[INFO] Using neural options: {neural_options.active}")

    # 3. Use default mutation weights
    mutation_weights = MutationWeights()
    print("[INFO] Using default mutation weights")

    # 4. Create dataset settings - simple Feynman equation with small sample size
    dataset_settings = DatasetSettings(
        dataset_name='feynman',
        num_samples=100,  # Small sample size for quick test
        rel_noise_magn=0.0001,
        eq_idx=10  # Use equation 10 (arbitrary choice)
    )
    print(f"[INFO] Dataset: {dataset_settings.dataset_name}, equation {dataset_settings.eq_idx}, {dataset_settings.num_samples} samples")

    # 5. Initialize the packaged model
    print("[INFO] Initializing PySR model...")
    packaged_model = init_pysr_model(
        model_settings=model_settings,
        neural_options=neural_options,
        mutation_weights=mutation_weights
    )

    # 6. Set up test log directory
    log_dir = Path(__file__).parent / 'test_logs' / 'minimal_test'
    print(f"[INFO] Log directory: {log_dir}")

    # 7. Run the test
    print("[INFO] Starting run_single test...")
    try:
        run_single(
            packaged_model=packaged_model,
            dataset_settings=dataset_settings,
            log_dir=str(log_dir),
            wandb_logging=False,  # Disable WandB for minimal test
            enable_mutation_logging=True
        )
        print("[SUCCESS] run_single completed successfully!")
        print(f"[INFO] Results saved to: {log_dir}")

    except Exception as e:
        print(f"[ERROR] run_single failed: {e}")
        raise

if __name__ == '__main__':
    print("=" * 50)
    print("MINIMAL TEST SCRIPT FOR run_single")
    print("=" * 50)

    test_run_single_minimal()

    print("=" * 50)
    print("TEST COMPLETED")
    print("=" * 50)