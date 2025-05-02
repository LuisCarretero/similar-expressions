from pathlib import Path

from run.benchmarking_utils import (
    NeuralOptions,
    MutationWeights,
    ModelSettings,
    DatasetSettings,
    init_pysr_model,
    run_single
)


def main():
    """
    Test the benchmarking utils.
    """
    log_basedir = Path('/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/round2')

    # Model settings
    model_settings = ModelSettings(
        niterations=10
    )
    neural_options = NeuralOptions(
        active=True,
        model_path='/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-e51hcsb9.onnx',
        device='cuda',
    )
    mutation_weights = MutationWeights(
        weight_neural_mutate_tree=1.0,
    )

    # Create model
    model = init_pysr_model(model_settings, mutation_weights, neural_options)

    # Run benchmark
    for eq_idx in [1, 2]:
        dataset_settings = DatasetSettings(
            dataset_name='synthetic',
            eq_idx=eq_idx,
            num_samples=2000,
            noise=1e-4,
        )

        run_single(
            model, 
            dataset_settings, 
            log_dir=str(log_basedir / f'eq{eq_idx}')
        )


if __name__ == '__main__':
    main()