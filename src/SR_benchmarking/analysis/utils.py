
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any


def load_mutations_data(path_logdir: str) -> pd.DataFrame:
    """
    Loads the mutations data from the given log directory and returns a dataframe of the mutations data.
    Each line in the dataframe corresponds to a single mutation executed during the SR run.
    """
    fpath = list(Path(path_logdir).glob('mutations*.csv'))
    if len(fpath) == 0:
        raise FileNotFoundError(f"No mutations data found in {path_logdir}")
    if len(fpath) > 1:
        print(f"Found multiple mutations data files in {path_logdir}, using the first one: {fpath[0]}")
    return pd.read_csv(fpath[0])

def load_neural_stats(path_logdir: str) -> dict:
    """
    Loads the neural stats from the given log directory and returns a dictionary of the neural stats.
    Note that these stats are summaries of the neural mutation process and not temporally resolved.
    """
    fpath = os.path.join(path_logdir, 'neural_stats.json')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No neural stats found in {path_logdir}")
    with open(fpath, 'r') as f:
        dct = json.load(f)
    return dct

def load_tensorboard_data(log_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the tensorboard data from the given log directory and returns a tuple of two dataframes:
    - df_scalars: a dataframe of all scalar values logged in the tensorboard
    - df_exprs: a dataframe of the expression values logged in the tensorboard
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    event_tags = event_acc.Tags()

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
    df_scalars, _ = load_tensorboard_data(log_dir)

    def get_prefixed_key(k: str) -> str:
        if k.startswith('loss'):
            return f'complexity_losses/{k}'
        return k

    for _, row in df_scalars.iterrows():
        row = row.to_dict()
        step = int(row['step'])
        values = {get_prefixed_key(k): v for k, v in row.items() if k != 'step'}
        step_stats.append((step, values))

    # Mutation stats
    CATEGORY_COL, LOSS_RATIO_THRESHOLD = 'mutation_type', 2

    df = load_mutations_data(log_dir)
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
        

    # Neural stats
    MUTATE_KEYS = [
        'total_samples', 'tree_build_failures', 'tree_comparison_failures', 'encoding_failures', 'decoding_failures', 
        'expr_similarity_failures', 'orig_tree_eval_failures', 'new_tree_eval_failures', 'skeleton_not_novel', 
        'multivariate_decoding_attempts'
    ]

    neural_stats = load_neural_stats(log_dir)
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
        step_stats_df[f'{combined_prefix}{key}'] = step_stats_df.loc[:, step_stats_df.columns.str.endswith(key)].mean(axis=1)

    # Combine summary stats
    summary_stats_series = pd.concat(summary_stats_series, axis=0)
    unique_row_names = list(set(map(lambda x: '/'.join(x.split('/')[1:]), summary_stats_series.index)))
    for key in unique_row_names:
        summary_stats_series[f'{combined_prefix}{key}'] = summary_stats_series.loc[summary_stats_series.index.str.endswith(key)].mean()

    # Remove single runs if requested
    if not keep_single_runs:
        step_stats_df = step_stats_df.loc[:, step_stats_df.columns.str.startswith(combined_prefix)]
        summary_stats_series = summary_stats_series.loc[summary_stats_series.index.str.startswith(combined_prefix)]

    return [(step, values.to_dict()) for step, values in step_stats_df.iterrows()], summary_stats_series.to_dict()