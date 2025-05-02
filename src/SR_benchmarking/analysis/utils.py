
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import json
from pathlib import Path


def load_mutations_data(path_logdir: str) -> pd.DataFrame:
    """
    Loads the mutations data from the given log directory and returns a dataframe of the mutations data.
    Each line in the dataframe corresponds to a single mutation executed during the SR run.
    """
    fpath = next(Path(path_logdir).glob('mutations*.csv'))
    return pd.read_csv(fpath)

def load_neural_stats(path_logdir: str) -> dict:
    """
    Loads the neural stats from the given log directory and returns a dictionary of the neural stats.
    Note that these stats are summaries of the neural mutation process and not temporally resolved.
    """
    fpath = os.path.join(path_logdir, 'neural_stats.json')
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
        df_scalars = pd.merge(df_scalars, new_df, on=['wall_time', 'step'], how='outer')

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

        return pd.DataFrame(tensors, columns=['wall_time', 'step', tag])

    df_exprs = expr_tensor_events_to_df(event_tags['tensors'][0])
    for tensor_tag in event_tags['tensors'][1:]:
        new_df = expr_tensor_events_to_df(tensor_tag)
        df_exprs = pd.merge(df_exprs, new_df, on=['timestamp', 'step'], how='outer')
    
    assert len(df_exprs) == df_exprs.step.nunique() == df_exprs.timestamp.nunique(), "Merged dataframe has duplicate times/steps"
    df_exprs['timestamp'] = df_exprs['timestamp'].astype(float)
    
    return df_scalars, df_exprs
