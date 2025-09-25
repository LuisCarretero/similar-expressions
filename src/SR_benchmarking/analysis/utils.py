
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any


def load_mutations_data(path_logdir: str) -> pd.DataFrame | None:
    """
    Loads the mutations data from the given log directory and returns a dataframe of the mutations data.
    Each line in the dataframe corresponds to a single mutation executed during the SR run.
    Returns None if no mutations data is found (when mutation logging is disabled).
    """
    fpath = list(Path(path_logdir).glob('mutations*.csv'))
    if len(fpath) == 0:
        return None
    if len(fpath) > 1:
        print(f"Found multiple mutations data files in {path_logdir}, using the first one: {fpath[0]}")
    return pd.read_csv(fpath[0])

def load_neural_stats(path_logdir: str) -> dict | None:
    """
    Loads the neural stats from the given log directory and returns a dictionary of the neural stats.
    Note that these stats are summaries of the neural mutation process and not temporally resolved.
    Returns None if no neural stats file is found.
    """
    fpath = os.path.join(path_logdir, 'neural_stats.json')
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r') as f:
        dct = json.load(f)
    return dct

def load_tensorboard_data(log_dir: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Loads the tensorboard data from the given log directory and returns a tuple of two dataframes:
    - df_scalars: a dataframe of all scalar values logged in the tensorboard
    - df_exprs: a dataframe of the expression values logged in the tensorboard
    Returns None if no tensorboard data is found.
    """
    if not os.path.exists(log_dir):
        return None
    
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    event_tags = event_acc.Tags()
    
    # Check if there are any scalars or tensors
    if 'scalars' not in event_tags or len(event_tags['scalars']) == 0:
        return None

    # Scalars
    def scalar_events_to_df(tag: str) -> pd.DataFrame:
        scalars = np.array([(e.wall_time, e.step, e.value) for e in event_acc.Scalars(tag)])

        if 'complexity' not in tag:
            tag = tag.split('/')[-1]
        else:
            tag = '_'.join(tag.split('complexity=')[-1].split('/')[::-1])

        return pd.DataFrame(scalars, columns=['timestamp', 'step', tag])

    df_scalars = scalar_events_to_df(event_tags['scalars'][0])
    for scalar_tag in event_tags['scalars'][1:]:
        new_df = scalar_events_to_df(scalar_tag)
        df_scalars = pd.merge(df_scalars, new_df, on=['timestamp', 'step'], how='outer')

    assert len(df_scalars) == df_scalars.step.nunique() == df_scalars.timestamp.nunique(), "Merged dataframe has duplicate times/steps"
    df_scalars['step'] = df_scalars['step'].astype(int)
    # Sort columns that match the pattern 'loss_i' by their index i
    loss_cols = [col for col in df_scalars.columns if col.startswith('loss_') and col[5:].isdigit()]
    loss_cols_sorted = sorted(loss_cols, key=lambda x: int(x.split('_')[1]))
    other_cols = [col for col in df_scalars.columns if col not in loss_cols]
    df_scalars = df_scalars[other_cols + loss_cols_sorted]
    
    # Tensors
    def expr_tensor_events_to_df(tag: str) -> pd.DataFrame:
        tensors = np.array([
            (e.wall_time, e.step, e.tensor_proto.string_val[0].decode('utf-8').strip('"')) 
            for e in event_acc.Tensors(tag)
        ])

        if 'complexity' not in tag:
            tag = tag.split('/')[-1]
        else:
            tag = '_'.join(tag.split('complexity=')[-1].split('/')[::-1])

        return pd.DataFrame(tensors, columns=['timestamp', 'step', tag])

    df_exprs = expr_tensor_events_to_df(event_tags['tensors'][0])
    for tensor_tag in event_tags['tensors'][1:]:
        new_df = expr_tensor_events_to_df(tensor_tag)
        df_exprs = pd.merge(df_exprs, new_df, on=['timestamp', 'step'], how='outer')
    
    assert len(df_exprs) == df_exprs.step.nunique() == df_exprs.timestamp.nunique(), "Merged dataframe has duplicate times/steps"
    df_exprs['timestamp'] = df_exprs['timestamp'].astype(float)
    
    return df_scalars, df_exprs

def get_run_data_for_wandb(
    log_dir: str,
) -> Tuple[
    List[Tuple[int, Dict[str, Any]]],  # step_stats
    Dict[str, Any]  # summary_stats
]:
    """
    Get the run data from the log directory. This function extracts both step-by-step metrics and summary statistics
    from a PySR run's log directory. It processes tensorboard scalar data, mutation statistics, and neural network
    performance metrics, formatting them for easy logging to WandB.
    Returns:
        step_stats: List[ Tuple[int, Dict[str, Any]] ] - A list of tuples where each tuple contains:
            - An integer representing the step/iteration number
            - A dictionary mapping metric names to their values at that step
        summary_stats: Dict[str, Any] - A dictionary containing summary statistics for the entire run,
            including aggregated mutation and neural network performance metrics
    """

    step_stats = []
    summary_stats = {}

    # General logs
    tensorboard_data = load_tensorboard_data(log_dir)
    if tensorboard_data is None:
        raise FileNotFoundError(f"No tensorboard data found in {log_dir}")
    df_scalars, _ = tensorboard_data

    def get_prefixed_key(k: str) -> str:
        if k.startswith('loss'):
            return f'complexity_losses/{k}'
        return k

    for _, row in df_scalars.iterrows():
        row = row.to_dict()
        step = int(row['step'])
        values = {get_prefixed_key(k): v for k, v in row.items() if k != 'step'}
        step_stats.append((step, values))

    # Mutation stats (optional - only if mutations data is available)
    df = load_mutations_data(log_dir)
    if df is not None:
        CATEGORY_COL, LOSS_RATIO_THRESHOLD = 'mutation_type', 2
        df['loss_ratio'] = df['loss_after'] / df['loss_before']
        TED_df = df[[CATEGORY_COL, 'TED']].groupby(CATEGORY_COL).TED.mean().round(2)
        summary_stats.update({
            f"mutation_stats/TED/{cat}": mean
            for cat, mean in TED_df.items()
        })

        perc_loss_df = df[[CATEGORY_COL, 'loss_ratio']].groupby(CATEGORY_COL).apply(lambda x: (x < LOSS_RATIO_THRESHOLD).mean() * 100).round(1)['loss_ratio']
        summary_stats.update({
            f"mutation_stats/perc_<{LOSS_RATIO_THRESHOLD}/{cat}": mean 
            for cat, mean in perc_loss_df.items()
        })

    neural_stats = load_neural_stats(log_dir)
    if neural_stats is None:
        raise FileNotFoundError(f"No neural stats found in {log_dir}")
    
    # Neural stats
    MUTATE_KEYS = [
        'total_samples', 'tree_build_failures', 'tree_comparison_failures', 'encoding_failures', 'decoding_failures', 
        'expr_similarity_failures', 'orig_tree_eval_failures', 'new_tree_eval_failures', 'skeleton_not_novel', 
        'multivariate_decoding_attempts'
    ]

    count_stats = {k: v for k, v in neural_stats.items() if k.split('_')[-1] not in ['mean', 'std', 'ninvalid']}
    get_group = lambda k: "mutate" if k in MUTATE_KEYS else "sampling"
    summary_stats.update({
        f"neural_stats/{get_group(k)}/{k}": float(v)
        for k, v in count_stats.items()
    })

    return step_stats, summary_stats

def collect_sweep_results(
    log_dir: str,
    run_names: List[str],
    keep_single_runs: bool = False,
    combined_prefix: str = 'mean/'
) -> Tuple[
    List[Tuple[int, Dict[str, Any]]],  # step_stats
    Dict[str, Any]  # summary_stats
]:
    """
    Collect step and summary stats from multiple runs and combines them. Returns data in the same 
    format as get_run_data_for_wandb.
    """
    step_stats_dfs, summary_stats_series = [], []
    for run_name in run_names:
        dirpath = os.path.join(log_dir, run_name)
        step_stats, summary_stats = get_run_data_for_wandb(dirpath)

        # Step stats
        df = pd.DataFrame([values for _, values in step_stats], index=[step for step, _ in step_stats])
        df.columns = [f"{run_name}/{col}" for col in df.columns]
        step_stats_dfs.append(df)

        # Summary stats
        s = pd.Series(summary_stats)
        s.index = [f"{run_name}/{i}" for i in s.index]
        summary_stats_series.append(s)
        
    # Combine step stats
    step_stats_df = pd.concat(step_stats_dfs, axis=1)
    unique_col_names = list(set(map(lambda x: '/'.join(x.split('/')[1:]), step_stats_df.columns)))
    for key in unique_col_names:
        step_stats_df[f'{combined_prefix}{key}'] = step_stats_df.loc[:, step_stats_df.columns.str.endswith(key)].mean(axis=1, skipna=True)

    # Combine summary stats
    summary_stats_series = pd.concat(summary_stats_series, axis=0)
    unique_row_names = list(set(map(lambda x: '/'.join(x.split('/')[1:]), summary_stats_series.index)))
    for key in unique_row_names:
        summary_stats_series[f'{combined_prefix}{key}'] = summary_stats_series.loc[summary_stats_series.index.str.endswith(key)].mean(skipna=True)

    # Remove single runs if requested
    if not keep_single_runs:
        step_stats_df = step_stats_df.loc[:, step_stats_df.columns.str.startswith(combined_prefix)]
        summary_stats_series = summary_stats_series.loc[summary_stats_series.index.str.startswith(combined_prefix)]

    return [(step, values.to_dict()) for step, values in step_stats_df.iterrows()], summary_stats_series.to_dict()


# Pareto volume calculation functions
# Based on SymbolicRegression.jl implementation

def _convex_hull(xy: np.ndarray) -> np.ndarray:
    """Uses gift wrapping algorithm to create a convex hull."""
    assert xy.shape[1] == 2
    if len(xy) < 3:
        return xy

    cur_point = xy[np.argmin(xy[:, 0])]
    hull = []

    while True:
        hull.append(cur_point.copy())
        end_point = xy[0].copy()

        for candidate_point in xy:
            if np.array_equal(end_point, cur_point) or _is_left_of(candidate_point, (cur_point, end_point)):
                end_point = candidate_point.copy()

        cur_point = end_point.copy()
        if len(hull) > 1 and np.array_equal(end_point, hull[0]):
            break
        if len(hull) > len(xy):  # Prevent infinite loops
            break

    return np.array(hull)


def _is_left_of(point: np.ndarray, line: tuple) -> bool:
    """Check if point is to the left of the line."""
    start_point, end_point = line
    return ((end_point[0] - start_point[0]) * (point[1] - start_point[1]) -
            (end_point[1] - start_point[1]) * (point[0] - start_point[0])) > 0


def _convex_hull_area(hull: np.ndarray) -> float:
    """Computes area within convex hull using vectorized shoelace formula."""
    if len(hull) < 3:
        return 0.0

    x, y = hull[:, 0], hull[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    area = np.sum(x * y_next - x_next * y)
    return abs(area) / 2.0


def _extract_pareto_frontier_from_row(row: pd.Series) -> tuple:
    """Extract losses and complexities from dataframe row."""
    loss_cols = [col for col in row.index if col.startswith('loss_') and col[5:].isdigit()]
    if not loss_cols:
        return np.array([]), np.array([])

    loss_values = row[loss_cols].values
    complexities = np.array([int(col.split('_')[1]) for col in loss_cols])
    valid_mask = ~pd.isna(loss_values)

    return loss_values[valid_mask], complexities[valid_mask]


def pareto_volume(losses: np.ndarray, complexities: np.ndarray, maxsize: int, use_linear_scaling: bool = False) -> float:
    """
    Calculate pareto volume using convex hull approach.

    Args:
        losses: Array of loss values from Pareto frontier
        complexities: Corresponding complexity values
        maxsize: Maximum allowed complexity
        use_linear_scaling: Whether to use linear (True) or log (False) scaling for losses

    Returns:
        Pareto volume as float
    """
    if len(losses) == 0:
        return 0.0

    # Apply scaling transformations
    y = losses.copy() if use_linear_scaling else np.log10(losses + np.finfo(float).eps)
    x = np.log10(complexities)

    # Add anchor points
    min_y, max_y, max_x = np.min(y), np.max(y), np.max(x)
    y = np.append(y, [min_y, max_y])
    x = np.append(x, [np.log10(maxsize + 1), max_x])

    # Compute convex hull and area
    xy = np.column_stack((x, y))
    hull = _convex_hull(xy)
    return _convex_hull_area(hull)


def calculate_pareto_volume_from_row(row: pd.Series, maxsize: int = 30, use_linear_scaling: bool = False) -> float:
    """
    Calculate pareto volume from a single dataframe row for use with df.apply().

    Args:
        row: Pandas Series with loss_i columns
        maxsize: Maximum complexity to consider (default 30)
        use_linear_scaling: Whether to use linear scaling (default False for log scaling)

    Returns:
        Calculated pareto volume
    """
    losses, complexities = _extract_pareto_frontier_from_row(row)
    if len(losses) == 0:
        return 0.0
    return pareto_volume(losses, complexities, maxsize, use_linear_scaling)