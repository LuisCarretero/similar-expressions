# Layer 1: SR Execution (`sr_runner.py`)

## Purpose
Pure SR execution logic. No knowledge of Optuna or distributed coordination.

## Design

```python
class SRBatchRunner:
    """Runs SR experiments. Initialized ONCE per trial with fixed hyperparameters."""

    def __init__(self, model_settings, neural_options, mutation_weights,
                 dataset_config, trial_id):
        """
        Initialize runner with hyperparameters.

        Key: PySR model is created ONCE here and reused for all equations.
        """
        # Initialize PySR model ONCE
        self.packaged_model = init_pysr_model(model_settings, mutation_weights, neural_options)
        self.dataset_config = dataset_config
        self.trial_id = trial_id
        self.trial_dir = base_dir / f"trial_{trial_id}"

    def run_equations(self, equations: List[int], interruption_flag) -> List[float]:
        """
        Run a list of equations and return their PVs (in order).

        Supports resumption: scans for existing results and only runs remaining.
        """
        # Scan for existing results (trial-level resumption)
        existing_results = self._scan_existing_results(equations)

        # Determine remaining equations
        remaining = [eq for eq in equations if eq not in existing_results]

        all_results_map = existing_results.copy()

        # Run remaining equations
        for eq_idx in remaining:
            if interruption_flag():
                break
            pv = self._run_single_equation(eq_idx)
            all_results_map[eq_idx] = pv

        # Return in original order (0.0 for any interrupted/failed)
        return [all_results_map.get(eq, 0.0) for eq in equations]

    def _run_single_equation(self, eq_idx) -> float:
        """
        Run n_runs for one equation, return mean PV.

        Handles run-level resumption: reuses existing completed runs.
        """
        eq_pareto_volumes = []
        n_runs = self.dataset_config['n_runs']
        dataset_name = self.dataset_config['name']

        for run_i in range(n_runs):
            run_dir = self.trial_dir / f"{dataset_name}_eq{eq_idx}_run{run_i}"

            # Skip if already completed
            results_csv = run_dir / 'tensorboard_scalars.csv'
            if results_csv.exists():
                pv = self._extract_run_pv(run_dir)
                eq_pareto_volumes.append(pv)
                continue

            # Clean incomplete runs
            if run_dir.exists():
                shutil.rmtree(run_dir)

            # Create dataset and run
            dataset_settings = DatasetSettings(
                dataset_name=dataset_name,
                eq_idx=eq_idx,
                num_samples=self.dataset_config['num_samples'],
                rel_noise_magn=self.dataset_config['noise'],
                remove_op_equations=self.dataset_config.get('remove_op_equations', {})
            )

            run_single(
                packaged_model=self.packaged_model,
                dataset_settings=dataset_settings,
                log_dir=str(run_dir),
                wandb_logging=False,
                enable_mutation_logging=False
            )

            pv = self._extract_run_pv(run_dir)
            eq_pareto_volumes.append(pv)

        return np.mean(eq_pareto_volumes)

    def _scan_existing_results(self, equations) -> Dict[int, float]:
        """
        Scan trial directory for completed runs.

        Only includes equations where ALL n_runs are complete.
        """
        if not self.trial_dir.exists():
            return {}

        existing_results = {}
        n_runs = self.dataset_config['n_runs']
        dataset_name = self.dataset_config['name']

        for eq_idx in equations:
            eq_pareto_volumes = []
            all_runs_complete = True

            for run_i in range(n_runs):
                run_dir = self.trial_dir / f"{dataset_name}_eq{eq_idx}_run{run_i}"
                results_csv = run_dir / 'tensorboard_scalars.csv'

                if results_csv.exists():
                    try:
                        pv = self._extract_run_pv(run_dir)
                        eq_pareto_volumes.append(pv)
                    except Exception:
                        all_runs_complete = False
                        break
                else:
                    all_runs_complete = False
                    break

            if all_runs_complete and len(eq_pareto_volumes) == n_runs:
                eq_mean_pv = np.mean(eq_pareto_volumes)
                existing_results[eq_idx] = eq_mean_pv

        return existing_results

    def _extract_run_pv(self, run_dir) -> float:
        """Extract PV from tensorboard CSV."""
        csv_path = run_dir / 'tensorboard_scalars.csv'
        df = pd.read_csv(csv_path)
        final_pareto_volume = df['pareto_volume'].iloc[-1]
        return float(final_pareto_volume)
```

## Key Properties

- **Stateful**: Created once per trial with hyperparameters
- **Resumable**: Scans for existing results at equation and run level
- **Pure SR**: No Optuna or coordination concerns
- **Testable**: Can be tested independently with mock configs
