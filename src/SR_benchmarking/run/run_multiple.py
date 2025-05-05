from pathlib import Path
import argparse
from typing import List


from run.benchmarking_utils import (
    NeuralOptions,
    MutationWeights,
    ModelSettings,
    DatasetSettings,
    init_pysr_model,
    run_single
)


def main(
    equations: List[int], 
    dataset: str, 
    log_dir: str,
    pysr_verbosity: int = 0,
) -> None:
    """
    Test the benchmarking utils.
    """
    

    # Model settings
    model_settings = ModelSettings(
        niterations=10,
        verbosity=pysr_verbosity,
    )
    neural_options = NeuralOptions(
        active=True,
        model_path='/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-e51hcsb9.onnx',
        device='cuda',
    )
    mutation_weights = MutationWeights(
        weight_neural_mutate_tree=1.0,
        weight_mutate_constant = 0.0353,
        weight_mutate_operator = 3.63,
        weight_swap_operands = 0.00608,
        weight_rotate_tree = 1.42,
        weight_add_node = 0.0771,
        weight_insert_node = 2.44,
        weight_delete_node = 0.369,
        weight_simplify = 0.00148,
        weight_randomize = 0.00695,
        weight_do_nothing = 0.431,
        weight_optimize = 0.0,
    )

    # Create model
    model = init_pysr_model(model_settings, mutation_weights, neural_options)

    # Run benchmark
    for eq_idx in equations:
        print(f'[INFO] Running equation {eq_idx} from dataset {dataset}')
        dataset_settings = DatasetSettings(
            dataset_name=dataset,
            eq_idx=eq_idx,
            num_samples=2000,
            noise=1e-4,
        )

        run_single(
            model, 
            dataset_settings,
            log_dir=str(Path(log_dir) / f'{dataset}_eq{eq_idx}')
        )

def str_to_list(s: str) -> List[int]:
    """
    Convert a string to a slice.
    """
    if ':' in s:
        return list(range(*map(lambda x: int(x) if x else None, s.split(':'))))
    else:
        return [int(s)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--equations', type=str_to_list, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/variance_tests')
    parser.add_argument('--pysr_verbosity', type=int, default=0)
    args = parser.parse_args()

    main(
        equations=args.equations, 
        dataset=args.dataset, 
        log_dir=args.log_dir,
        pysr_verbosity=args.pysr_verbosity,
    )