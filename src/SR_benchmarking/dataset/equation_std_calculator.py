"""
Calculate standard deviation of equations in PySR-univariate.csv by evaluating them
over a uniform grid in the interval [a, b].
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Optional

# Import mathematical functions needed for evaluation (same as in utils.py)
from math import pi, cos, sin, sqrt, exp, log, tanh, sinh, cosh
arcsin = np.arcsin
arccos = np.arccos
ln = log  # Natural log alias


def evaluate_equation_on_grid(equation: str, a: float = 0, b: float = 10, num_points: int = 1000) -> Tuple[np.ndarray, Optional[str]]:
    """
    Evaluate a univariate equation over a uniform grid.

    Args:
        equation: Equation string in format "y = expression"
        a: Start of interval
        b: End of interval
        num_points: Number of points in the grid

    Returns:
        Tuple of (evaluated_values, error_message)
        If evaluation fails, returns (empty_array, error_message)
    """
    try:
        expr = equation.strip()

        # Replace ^ with ** for Python evaluation (following utils.py pattern)
        expr = expr.replace("^", "**")

        # Create uniform grid, avoiding x=0 for expressions that might divide by zero
        # Add small epsilon to avoid potential division by zero issues
        epsilon = 1e-10
        x_values = np.linspace(max(a, epsilon), b, num_points)

        # Create lambda function for evaluation
        # Using same pattern as in utils.py sample_equation function
        expr_func = eval(f"lambda x: {expr}")

        # Evaluate over the grid
        y_values = np.array([expr_func(x) for x in x_values])

        # Check for invalid values (NaN, inf)
        if np.any(~np.isfinite(y_values)):
            finite_mask = np.isfinite(y_values)
            if np.sum(finite_mask) < len(y_values) * 0.5:  # Less than 50% valid values
                return np.array([]), f"Too many invalid values: {np.sum(~finite_mask)}/{len(y_values)}"
            else:
                y_values = y_values[finite_mask]
                warnings.warn(f"Removed {np.sum(~finite_mask)} invalid values from evaluation")

        return y_values, None

    except Exception as e:
        return np.array([]), f"Evaluation error: {str(e)}"


def calculate_equation_statistics(equation: str, a: float = 0, b: float = 10, num_points: int = 1000) -> dict:
    """
    Calculate statistics (mean, std, min, max) for an equation evaluated over a grid.

    Args:
        equation: Equation string
        a: Start of interval
        b: End of interval
        num_points: Number of evaluation points

    Returns:
        Dictionary with statistics and metadata
    """
    y_values, error = evaluate_equation_on_grid(equation, a, b, num_points)

    result = {
        'equation': equation,
        'evaluation_error': error
    }

    if len(y_values) > 0:
        result.update({
            'std': float(np.std(y_values)),
            'mean': float(np.mean(y_values)),
            'min': float(np.min(y_values)),
            'max': float(np.max(y_values))
        })
    else:
        result.update({
            'std': np.nan,
            'mean': np.nan,
            'min': np.nan,
            'max': np.nan
        })

    return result


def process_univariate_csv(input_csv_path: str, output_csv_path: str,
                          a: float = 0, b: float = 10, num_points: int = 1000) -> pd.DataFrame:
    """
    Process PySR-univariate.csv to calculate standard deviations.

    Args:
        input_csv_path: Path to input CSV
        output_csv_path: Path to save results
        a: Interval start
        b: Interval end
        num_points: Number of evaluation points

    Returns:
        DataFrame with results
    """
    print(f"Loading {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    print(f"Found {len(df)} equations")
    print(f"Evaluating over interval [{a}, {b}] with {num_points} points...")

    results = []

    for idx, row in df.iterrows():
        equation = row['true_equation']

        # Calculate statistics for original equation
        stats = calculate_equation_statistics(equation, a, b, num_points)

        # Add original CSV columns
        result = {
            'row_index': idx,
            'true_equation': equation,
            'equation_simplified': row.get('equation_simplified', ''),
            'complexity_multivar': row.get('complexity_multivar', ''),
            'complexity_simplified': row.get('complexity_simplified', ''),
            'eval_std': stats['std'],
            'eval_mean': stats['mean'],
            'eval_min_value': stats['min'],
            'eval_max_value': stats['max'],
            'evaluation_error': stats['evaluation_error']
        }

        results.append(result)

        # Progress indicator
        if (idx + 1) % 100 == 0 or idx < 10:
            print(f"Processed {idx + 1}/{len(df)} equations...")
            if stats['evaluation_error']:
                print(f"  Error in equation {idx}: {stats['evaluation_error']}")
            else:
                print(f"  Std: {stats['std']:.4f}, Mean: {stats['mean']:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")

    # Print summary statistics
    print_summary_statistics(results_df)

    return results_df


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for the processed results."""

    total_equations = len(df)
    successful_evaluations = df['evaluation_error'].isna().sum()
    failed_evaluations = total_equations - successful_evaluations

    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total equations: {total_equations}")
    print(f"Successful evaluations: {successful_evaluations} ({successful_evaluations/total_equations*100:.1f}%)")
    print(f"Failed evaluations: {failed_evaluations} ({failed_evaluations/total_equations*100:.1f}%)")

    # Statistics for successful evaluations
    successful_df = df[df['evaluation_error'].isna()]
    if not successful_df.empty:
        print(f"\nStatistics for successful evaluations:")
        print(f"Standard deviation - Mean: {successful_df['eval_std'].mean():.4f}, Median: {successful_df['eval_std'].median():.4f}")
        print(f"Standard deviation - Min: {successful_df['eval_std'].min():.4f}, Max: {successful_df['eval_std'].max():.4f}")
        print(f"Mean value - Mean: {successful_df['eval_mean'].mean():.4f}, Median: {successful_df['eval_mean'].median():.4f}")

        # Show equations with highest and lowest standard deviation
        print(f"\nTop 3 equations with highest std:")
        top_std = successful_df.nlargest(3, 'eval_std')
        for idx, row in top_std.iterrows():
            print(f"  Std: {row['eval_std']:.4f} - {row['true_equation'][:80]}...")

        print(f"\nTop 3 equations with lowest std:")
        low_std = successful_df.nsmallest(3, 'eval_std')
        for idx, row in low_std.iterrows():
            print(f"  Std: {row['eval_std']:.4f} - {row['true_equation'][:80]}...")

    # Show some failed equations if any
    failed_df = df[df['evaluation_error'].notna()]
    if not failed_df.empty:
        print(f"\nSample failed equations:")
        for idx, row in failed_df.head(3).iterrows():
            print(f"  Error: {row['evaluation_error']} - {row['true_equation'][:80]}...")


if __name__ == "__main__":
    # Configuration
    input_file = "/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/dataset/PySR-univariate.csv"
    output_file = "/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/dataset/PySR-univariate_std_results.csv"

    interval_start = 0
    interval_end = 10
    num_points = 1000

    print("="*80)
    print("EQUATION STANDARD DEVIATION CALCULATOR")
    print("="*80)

    # Process the dataset
    results = process_univariate_csv(
        input_csv_path=input_file,
        output_csv_path=output_file,
        a=interval_start,
        b=interval_end,
        num_points=num_points
    )

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)