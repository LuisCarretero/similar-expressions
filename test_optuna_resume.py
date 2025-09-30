"""
Test script to verify Optuna trial resumption strategy.

This script demonstrates:
1. Starting a trial and simulating interruption (saving state)
2. Restarting and reconstructing the trial with saved hyperparameters
3. Completing the trial and ensuring Optuna handles it correctly
"""

import json
import time
import optuna
from pathlib import Path


# Checkpoint file for saving/loading trial state
CHECKPOINT_FILE = Path("/tmp/optuna_resume_test_checkpoint.json")


def save_checkpoint(trial_number, params, completed_steps, intermediate_values):
    """Save trial state to checkpoint file."""
    checkpoint = {
        'trial_number': trial_number,
        'params': params,
        'completed_steps': completed_steps,
        'intermediate_values': intermediate_values
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"[CHECKPOINT] Saved: trial={trial_number}, steps={completed_steps}")


def load_checkpoint():
    """Load trial state from checkpoint file."""
    if not CHECKPOINT_FILE.exists():
        return None
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint = json.load(f)
    print(f"[CHECKPOINT] Loaded: trial={checkpoint['trial_number']}, steps={checkpoint['completed_steps']}")
    return checkpoint


def clear_checkpoint():
    """Remove checkpoint file."""
    CHECKPOINT_FILE.unlink(missing_ok=True)
    print("[CHECKPOINT] Cleared")


def objective(trial, allow_interruption=False):
    """
    Objective function that simulates a long-running computation with checkpointing.

    Simulates 10 steps, where each step takes time. We'll interrupt after step 3
    and resume from there.

    Args:
        allow_interruption: If True, allow simulated interruption at step 3
    """
    # Check for existing checkpoint
    checkpoint = load_checkpoint()

    if checkpoint is not None:
        # Resume from checkpoint
        print(f"\n[RESUME] Resuming trial {checkpoint['trial_number']}")
        print(f"[RESUME] Reconstructing with saved params: {checkpoint['params']}")

        # Manually set the trial parameters to match checkpoint
        # Note: We need to suggest the same parameters with the same names
        for param_name, param_value in checkpoint['params'].items():
            if isinstance(param_value, float):
                # Suggest with a tight range around the saved value
                trial.suggest_float(param_name, param_value, param_value)
            elif isinstance(param_value, int):
                trial.suggest_int(param_name, param_value, param_value)

        # Restore intermediate values
        for step, value in checkpoint['intermediate_values'].items():
            trial.report(float(value), int(step))

        start_step = checkpoint['completed_steps']
        print(f"[RESUME] Starting from step {start_step}")
    else:
        # Fresh trial
        print(f"\n[FRESH] Starting new trial {trial.number}")

        # Sample hyperparameters normally
        x = trial.suggest_float('x', 0.0, 10.0)
        y = trial.suggest_int('y', 0, 100)

        print(f"[FRESH] Sampled params: x={x:.4f}, y={y}")

        start_step = 0

    # Get current parameters (either sampled or reconstructed)
    x = trial.params['x']
    y = trial.params['y']

    # Simulate computation with intermediate reporting
    total_steps = 10
    intermediate_values = {}

    for step in range(start_step, total_steps):
        print(f"[TRIAL {trial.number}] Step {step}/{total_steps}...")

        # Simulate computation (simple quadratic)
        intermediate_value = -(x - 5.0)**2 - (y - 50.0)**2 / 100.0 + step * 0.1
        intermediate_values[step] = intermediate_value

        # Report intermediate value
        trial.report(intermediate_value, step)

        # Simulate work
        time.sleep(0.1)

        # Simulate interruption after step 3 (only when allowed and no checkpoint exists)
        if step == 3 and checkpoint is None and allow_interruption:
            print(f"\n[INTERRUPT] Simulating SLURM timeout at step {step}!")
            save_checkpoint(
                trial_number=trial.number,
                params=trial.params,
                completed_steps=step + 1,
                intermediate_values=intermediate_values
            )
            # Raise an exception to simulate job interruption
            raise RuntimeError("SIMULATED_INTERRUPTION")

        # Check if should prune
        if trial.should_prune():
            print(f"[PRUNE] Trial {trial.number} pruned at step {step}")
            raise optuna.TrialPruned()

    # Trial completed successfully
    final_value = intermediate_value
    print(f"[COMPLETE] Trial {trial.number} finished: {final_value:.4f}")

    # Clear checkpoint on successful completion
    clear_checkpoint()

    return final_value


def run_experiment():
    """Run the experiment with interruption and resumption."""
    print("="*60)
    print("Testing Optuna Trial Resumption Strategy")
    print("="*60)

    # Create study with persistent SQLite storage
    db_path = "/tmp/optuna_resume_test.db"
    storage_url = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction='maximize',
        storage=storage_url,
        study_name='resume_test',
        load_if_exists=True
    )
    print(f"Using database: {db_path}")

    # Clear any old checkpoint
    clear_checkpoint()

    print("\n" + "="*60)
    print("PHASE 1: Initial trial (will be interrupted)")
    print("="*60)

    # First attempt - will be interrupted
    try:
        study.optimize(lambda trial: objective(trial, allow_interruption=True), n_trials=1, catch=(RuntimeError,))
    except Exception as e:
        print(f"[ERROR] Caught exception: {e}")

    # Check study state after interruption
    print("\n" + "="*60)
    print("STUDY STATE AFTER INTERRUPTION")
    print("="*60)
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    if study.trials:
        last_trial = study.trials[-1]
        print(f"\nLast trial:")
        print(f"  Number: {last_trial.number}")
        print(f"  State: {last_trial.state}")
        print(f"  Params: {last_trial.params}")
        print(f"  Intermediate values: {last_trial.intermediate_values}")

    # Check if checkpoint exists
    assert CHECKPOINT_FILE.exists(), "Checkpoint should exist after interruption"

    print("\n" + "="*60)
    print("PHASE 2: Resume trial (will complete)")
    print("="*60)

    # Second attempt - resume and complete
    study.optimize(objective, n_trials=1)

    # Check study state after completion
    print("\n" + "="*60)
    print("STUDY STATE AFTER COMPLETION")
    print("="*60)
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    # Show all trials
    print("\nAll trials:")
    for trial in study.trials:
        print(f"  Trial {trial.number}: state={trial.state}, params={trial.params}, value={trial.value}")

    # Verify checkpoint is cleared
    assert not CHECKPOINT_FILE.exists(), "Checkpoint should be cleared after successful completion"

    print("\n" + "="*60)
    print("PHASE 3: Run a few more normal trials")
    print("="*60)

    # Run more trials normally
    study.optimize(objective, n_trials=3)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Total trials: {len(study.trials)}")

    # Show all trials
    print("\nAll trials summary:")
    for trial in study.trials:
        print(f"  Trial {trial.number}: state={trial.state}, value={trial.value}, params={trial.params}")

    print("\n" + "="*60)
    print("DATABASE INSPECTION")
    print("="*60)
    print(f"Database location: {db_path}")
    print("\nYou can inspect the database with:")
    print(f"  sqlite3 {db_path}")
    print(f"  optuna-dashboard sqlite:///{db_path}")
    print("\nUseful SQL queries:")
    print("  SELECT trial_id, number, state, value FROM trials;")
    print("  SELECT * FROM trial_params;")
    print("  SELECT * FROM trial_intermediate_values;")


if __name__ == "__main__":
    run_experiment()
