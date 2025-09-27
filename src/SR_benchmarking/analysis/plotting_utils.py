"""
Plotting utilities for SR benchmarking analysis.

This module contains functions to create various plots for comparing
neural vs vanilla setups in symbolic regression benchmarking.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats


def add_statistical_textbox(ax, neural_vals, vanilla_vals, metric_type='loss',
                           position=(0.05, 0.95), fontsize=8):
    """Add a text box with statistical comparison between neural and vanilla setups."""
    if len(neural_vals) == 0 or len(vanilla_vals) == 0:
        return

    neural_mean = neural_vals.mean()
    neural_median = neural_vals.median()
    vanilla_mean = vanilla_vals.mean()
    vanilla_median = vanilla_vals.median()

    # Perform Wilcoxon test
    try:
        stat, pval = stats.wilcoxon(neural_vals, vanilla_vals)

        if metric_type == 'loss':
            textstr = f'Neural: μ={neural_mean:.2e}, med={neural_median:.2e}\n'
            textstr += f'Vanilla: μ={vanilla_mean:.2e}, med={vanilla_median:.2e}\n'
        else:  # volume or other metrics
            textstr = f'Neural: μ={neural_mean:.4f}, med={neural_median:.4f}\n'
            textstr += f'Vanilla: μ={vanilla_mean:.4f}, med={vanilla_median:.4f}\n'
        textstr += f'p-value: {pval:.4f}'

        color = 'lightgreen' if metric_type == 'loss' else 'lightblue'
        props = dict(boxstyle='round', facecolor=color, alpha=0.8)
        ax.text(position[0], position[1], textstr, transform=ax.transAxes, fontsize=fontsize,
                verticalalignment='top', bbox=props)
    except Exception:
        pass


def plot_basic_distributions(filtered_df, summary_stats):
    """
    Create basic distribution histograms comparing neural and vanilla setups.

    Parameters:
    - filtered_df: DataFrame with individual run results
    - summary_stats: DataFrame with summary statistics by equation and setup

    Returns:
    - fig: matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Neural vs Vanilla Setup Distributions', fontsize=16)

    # Filter for neural and vanilla setups
    neural_df = filtered_df[filtered_df['setup'] == 'neural']
    vanilla_df = filtered_df[filtered_df['setup'] == 'vanilla']

    # Plot 1: Log Final Min Loss histogram
    ax1 = axes[0, 0]
    log_loss_neural = np.log10(neural_df['final_min_loss'])
    log_loss_vanilla = np.log10(vanilla_df['final_min_loss'])

    # Create common bins for log loss
    log_loss_min = min(log_loss_neural.min(), log_loss_vanilla.min())
    log_loss_max = max(log_loss_neural.max(), log_loss_vanilla.max())
    log_loss_bins = np.linspace(log_loss_min, log_loss_max, 30)

    ax1.hist(log_loss_neural, bins=log_loss_bins, alpha=0.5, color='skyblue',
             edgecolor='black', label='Neural')
    ax1.hist(log_loss_vanilla, bins=log_loss_bins, alpha=0.5, color='orange',
             edgecolor='black', label='Vanilla')
    ax1.set_xlabel('Log10(Final Min Loss)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Log Final Min Loss Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Pareto Volume histogram
    ax2 = axes[0, 1]
    pareto_min = min(neural_df['final_pareto_volume'].min(), vanilla_df['final_pareto_volume'].min())
    pareto_max = max(neural_df['final_pareto_volume'].max(), vanilla_df['final_pareto_volume'].max())
    pareto_bins = np.linspace(pareto_min, pareto_max, 30)

    ax2.hist(neural_df['final_pareto_volume'], bins=pareto_bins, alpha=0.5, color='skyblue',
             edgecolor='black', label='Neural')
    ax2.hist(vanilla_df['final_pareto_volume'], bins=pareto_bins, alpha=0.5, color='orange',
             edgecolor='black', label='Vanilla')
    ax2.set_xlabel('Final Pareto Volume')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Pareto Volume Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Final Min Loss means histogram
    ax3 = axes[1, 0]
    neural_means = summary_stats[summary_stats['setup'] == 'neural']['final_min_loss_mean']
    vanilla_means = summary_stats[summary_stats['setup'] == 'vanilla']['final_min_loss_mean']

    log_loss_neural_means = np.log10(neural_means)
    log_loss_vanilla_means = np.log10(vanilla_means)

    log_loss_means_min = min(log_loss_neural_means.min(), log_loss_vanilla_means.min())
    log_loss_means_max = max(log_loss_neural_means.max(), log_loss_vanilla_means.max())
    log_loss_means_bins = np.linspace(log_loss_means_min, log_loss_means_max, 20)

    ax3.hist(log_loss_neural_means, bins=log_loss_means_bins, alpha=0.5, color='skyblue',
             edgecolor='black', label='Neural')
    ax3.hist(log_loss_vanilla_means, bins=log_loss_means_bins, alpha=0.5, color='orange',
             edgecolor='black', label='Vanilla')
    ax3.set_xlabel('Log10(Final Min Loss Mean)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Log Final Min Loss Mean Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Pareto Volume means histogram
    ax4 = axes[1, 1]
    neural_pareto_means = summary_stats[summary_stats['setup'] == 'neural']['final_pareto_volume_mean']
    vanilla_pareto_means = summary_stats[summary_stats['setup'] == 'vanilla']['final_pareto_volume_mean']

    pareto_means_min = min(neural_pareto_means.min(), vanilla_pareto_means.min())
    pareto_means_max = max(neural_pareto_means.max(), vanilla_pareto_means.max())
    pareto_means_bins = np.linspace(pareto_means_min, pareto_means_max, 20)

    ax4.hist(neural_pareto_means, bins=pareto_means_bins, alpha=0.5, color='skyblue',
             edgecolor='black', label='Neural')
    ax4.hist(vanilla_pareto_means, bins=pareto_means_bins, alpha=0.5, color='orange',
             edgecolor='black', label='Vanilla')
    ax4.set_xlabel('Final Pareto Volume Mean')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Pareto Volume Mean Distribution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    return fig


def plot_comprehensive_comparison(filtered_df, summary_stats):
    """
    Create comprehensive comparison plots with violin plots and scatter plots.

    Parameters:
    - filtered_df: DataFrame with individual run results
    - summary_stats: DataFrame with summary statistics by equation and setup

    Returns:
    - fig: matplotlib figure object
    """
    plt.style.use('default')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Neural vs Vanilla Setups', fontsize=16, y=0.98)

    # Plot 1: Final Min Loss - Violin plot
    ax1 = axes[0, 0]
    log_loss_data = filtered_df.copy()
    log_loss_data['log_final_min_loss'] = np.log10(log_loss_data['final_min_loss'])

    sns.violinplot(data=log_loss_data, x='setup', y='log_final_min_loss', ax=ax1, inner='box')
    ax1.set_title('Final Min Loss Distribution')
    ax1.set_ylabel('Log10(Final Min Loss)')
    ax1.set_xlabel('Setup')

    # Add median values as text
    for i, setup in enumerate(['neural', 'vanilla']):
        setup_data = log_loss_data[log_loss_data['setup'] == setup]['log_final_min_loss']
        median_val = setup_data.median()
        ax1.text(i, median_val + 0.5, f'Median: {10**median_val:.2e}',
                 ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Plot 2: Final Pareto Volume - Violin plot
    ax2 = axes[0, 1]
    sns.violinplot(data=filtered_df, x='setup', y='final_pareto_volume', ax=ax2, inner='box')
    ax2.set_title('Final Pareto Volume Distribution')
    ax2.set_ylabel('Final Pareto Volume')
    ax2.set_xlabel('Setup')

    # Add median values as text
    for i, setup in enumerate(['neural', 'vanilla']):
        setup_data = filtered_df[filtered_df['setup'] == setup]['final_pareto_volume']
        median_val = setup_data.median()
        ax2.text(i, median_val + 0.1, f'Median: {median_val:.4f}',
                 ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Plot 3: Final Min Loss - Scatter plot by equation
    ax3 = axes[1, 0]
    loss_pivot = summary_stats.pivot_table(
        index=['dataset', 'eq'],
        columns='setup',
        values='final_min_loss_mean'
    ).reset_index()

    if 'neural' in loss_pivot.columns and 'vanilla' in loss_pivot.columns:
        ax3.scatter(loss_pivot['vanilla'], loss_pivot['neural'], alpha=0.7, s=50)
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        # Add diagonal line for reference
        min_val = min(loss_pivot['vanilla'].min(), loss_pivot['neural'].min())
        max_val = max(loss_pivot['vanilla'].max(), loss_pivot['neural'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal performance')

        ax3.set_xlabel('Vanilla Setup - Final Min Loss (log scale)')
        ax3.set_ylabel('Neural Setup - Final Min Loss (log scale)')
        ax3.set_title('Final Min Loss: Neural vs Vanilla (by equation)')

        # Add Wilcoxon test result
        loss_stat, loss_pval = stats.wilcoxon(loss_pivot['neural'], loss_pivot['vanilla'])
        ax3.text(0.05, 0.95, f'Wilcoxon test:\nstatistic={loss_stat:.2f}\np-value={loss_pval:.4f}',
                 transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax3.legend()

    # Plot 4: Final Pareto Volume - Scatter plot by equation
    ax4 = axes[1, 1]
    volume_pivot = summary_stats.pivot_table(
        index=['dataset', 'eq'],
        columns='setup',
        values='final_pareto_volume_mean'
    ).reset_index()

    if 'neural' in volume_pivot.columns and 'vanilla' in volume_pivot.columns:
        ax4.scatter(volume_pivot['vanilla'], volume_pivot['neural'], alpha=0.7, s=50)

        # Add diagonal line for reference
        min_val = min(volume_pivot['vanilla'].min(), volume_pivot['neural'].min())
        max_val = max(volume_pivot['vanilla'].max(), volume_pivot['neural'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal performance')

        ax4.set_xlabel('Vanilla Setup - Final Pareto Volume')
        ax4.set_ylabel('Neural Setup - Final Pareto Volume')
        ax4.set_title('Final Pareto Volume: Neural vs Vanilla (by equation)')

        # Add Wilcoxon test result
        volume_stat, volume_pval = stats.wilcoxon(volume_pivot['neural'], volume_pivot['vanilla'])
        ax4.text(0.05, 0.95, f'Wilcoxon test:\nstatistic={volume_stat:.2f}\np-value={volume_pval:.4f}',
                 transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax4.legend()

    # Add summary statistics to the figure
    neural_loss = filtered_df[filtered_df['setup'] == 'neural']['final_min_loss']
    vanilla_loss = filtered_df[filtered_df['setup'] == 'vanilla']['final_min_loss']
    neural_volume = filtered_df[filtered_df['setup'] == 'neural']['final_pareto_volume']
    vanilla_volume = filtered_df[filtered_df['setup'] == 'vanilla']['final_pareto_volume']

    summary_text = f"""COMPARISON SUMMARY
Total equations: {filtered_df['eq'].nunique()}
Runs per setup: Neural={len(neural_loss)}, Vanilla={len(vanilla_loss)}

Final Min Loss:
  Neural - Mean: {neural_loss.mean():.2e}, Median: {neural_loss.median():.2e}
  Vanilla - Mean: {vanilla_loss.mean():.2e}, Median: {vanilla_loss.median():.2e}

Final Pareto Volume:
  Neural - Mean: {neural_volume.mean():.6f}, Median: {neural_volume.median():.6f}
  Vanilla - Mean: {vanilla_volume.mean():.6f}, Median: {vanilla_volume.median():.6f}"""

    fig.text(0.02, 0.02, summary_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    return fig


def plot_step_wise_metrics(filtered_df):
    """
    Create step-wise metric comparison plots and ratio analysis.

    Parameters:
    - filtered_df: DataFrame with step-wise metrics

    Returns:
    - fig1, fig2: matplotlib figure objects for metrics and ratios
    """
    comparison_metrics = [
        'min_loss_step_100', 'max_pareto_volume_step_100',
        'min_loss_step_600', 'max_pareto_volume_step_600',
        'min_loss_step_929', 'max_pareto_volume_step_929'
    ]

    # Create metrics comparison figure
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(comparison_metrics):
        ax = axes[i]

        # Create pivot table for this metric
        metric_pivot = filtered_df.pivot_table(
            index='eq', columns='setup', values=metric, aggfunc='mean'
        ).dropna()

        if 'neural' in metric_pivot.columns and 'vanilla' in metric_pivot.columns:
            ax.scatter(metric_pivot['vanilla'], metric_pivot['neural'], alpha=0.7, s=50)

            # Add diagonal line for reference
            min_val = min(metric_pivot['vanilla'].min(), metric_pivot['neural'].min())
            max_val = max(metric_pivot['vanilla'].max(), metric_pivot['neural'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal performance')

            # Set log scale for loss metrics
            if 'loss' in metric:
                ax.set_xscale('log')
                ax.set_yscale('log')

            ax.set_xlabel(f'Vanilla Setup - {metric}')
            ax.set_ylabel(f'Neural Setup - {metric}')
            ax.set_title(f'{metric}: Neural vs Vanilla')

            # Add Wilcoxon test result
            try:
                stat, pval = stats.wilcoxon(metric_pivot['neural'], metric_pivot['vanilla'])
                ax.text(0.05, 0.95, f'Wilcoxon:\np={pval:.4f}',
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            except:
                pass

            ax.legend(fontsize=8)

    plt.tight_layout()

    # Create ratio plots figure
    fig2, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Extract step numbers and create ratio data
    steps = [100, 600, 929]
    loss_ratios_by_step = []
    volume_ratios_by_step = []

    for step in steps:
        loss_metric = f'min_loss_step_{step}'
        volume_metric = f'max_pareto_volume_step_{step}'

        # Create pivot tables
        loss_pivot = filtered_df.pivot_table(
            index='eq', columns='setup', values=loss_metric, aggfunc='mean'
        ).dropna()

        volume_pivot = filtered_df.pivot_table(
            index='eq', columns='setup', values=volume_metric, aggfunc='mean'
        ).dropna()

        if 'neural' in loss_pivot.columns and 'vanilla' in loss_pivot.columns:
            # Calculate ratios (Neural / Vanilla)
            epsilon = 1e-12
            loss_ratio = loss_pivot['neural'] / (loss_pivot['vanilla'] + epsilon)
            volume_ratio = volume_pivot['neural'] / (volume_pivot['vanilla'] + epsilon)

            loss_ratios_by_step.append(loss_ratio.values)
            volume_ratios_by_step.append(volume_ratio.values)

    # Plot loss ratios
    ax1 = axes[0]
    parts = ax1.violinplot(loss_ratios_by_step, positions=range(len(steps)),
                           showmeans=True, showmedians=True)

    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal performance (ratio=1)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss Ratio (Neural / Vanilla)')
    ax1.set_title('Loss Ratio Distribution Over Steps')
    ax1.set_yscale('log')
    ax1.set_xticks(range(len(steps)))
    ax1.set_xticklabels([f'Step {step}' for step in steps])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot volume ratios
    ax2 = axes[1]
    bp = ax2.boxplot(volume_ratios_by_step, positions=range(len(steps)),
                     patch_artist=True, showfliers=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal performance (ratio=1)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Volume Ratio (Neural / Vanilla)')
    ax2.set_title('Pareto Volume Ratio Distribution Over Steps\n(Box plot to handle outliers)')
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels([f'Step {step}' for step in steps])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add median values as text annotations
    for i, step in enumerate(steps):
        median_vol = np.median(volume_ratios_by_step[i])
        ax2.text(i, median_vol, f'{median_vol:.2f}',
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    return fig1, fig2


def categorize_difficulty(filtered_df):
    """
    Categorize equations by difficulty based on vanilla setup performance.

    Parameters:
    - filtered_df: DataFrame with results

    Returns:
    - filtered_df: DataFrame with added 'difficulty' column
    """
    vanilla_eq_performance = filtered_df[filtered_df['setup'] == 'vanilla'].groupby('eq')['final_min_loss'].mean()

    # Define difficulty thresholds (using quantiles)
    easy_threshold = vanilla_eq_performance.quantile(0.33)
    medium_threshold = vanilla_eq_performance.quantile(0.67)

    def get_difficulty(eq_id):
        avg_loss = vanilla_eq_performance[eq_id]
        if avg_loss <= easy_threshold:
            return 'Easy'
        elif avg_loss <= medium_threshold:
            return 'Medium'
        else:
            return 'Difficult'

    filtered_df['difficulty'] = filtered_df['eq'].apply(get_difficulty)
    return filtered_df


def plot_difficulty_analysis(filtered_df):
    """
    Create difficulty-based analysis plots.

    Parameters:
    - filtered_df: DataFrame with difficulty categorization

    Returns:
    - fig: matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    difficulties = ['Easy', 'Medium', 'Difficult']
    colors = ['green', 'orange', 'red']

    for i, difficulty in enumerate(difficulties):
        # Filter data for this difficulty
        diff_data = filtered_df[filtered_df['difficulty'] == difficulty]

        # Create pivot tables for this difficulty
        loss_pivot_diff = diff_data.pivot_table(
            index='eq', columns='setup', values='final_min_loss', aggfunc='mean'
        ).dropna()

        volume_pivot_diff = diff_data.pivot_table(
            index='eq', columns='setup', values='final_pareto_volume', aggfunc='mean'
        ).dropna()

        # Plot 1: Final Min Loss comparison
        ax1 = axes[0, i]
        if 'neural' in loss_pivot_diff.columns and 'vanilla' in loss_pivot_diff.columns:
            ax1.scatter(loss_pivot_diff['vanilla'], loss_pivot_diff['neural'],
                       alpha=0.7, s=50, color=colors[i])

            # Add diagonal line
            min_val = min(loss_pivot_diff['vanilla'].min(), loss_pivot_diff['neural'].min())
            max_val = max(loss_pivot_diff['vanilla'].max(), loss_pivot_diff['neural'].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

            # Add statistics
            add_statistical_textbox(ax1, loss_pivot_diff['neural'], loss_pivot_diff['vanilla'], 'loss')

        ax1.set_xlabel('Vanilla Setup - Final Min Loss')
        ax1.set_ylabel('Neural Setup - Final Min Loss')
        ax1.set_title(f'{difficulty} Equations - Final Min Loss')
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # Plot 2: Final Pareto Volume comparison
        ax2 = axes[1, i]
        if 'neural' in volume_pivot_diff.columns and 'vanilla' in volume_pivot_diff.columns:
            ax2.scatter(volume_pivot_diff['vanilla'], volume_pivot_diff['neural'],
                       alpha=0.7, s=50, color=colors[i])

            # Add diagonal line
            min_val = min(volume_pivot_diff['vanilla'].min(), volume_pivot_diff['neural'].min())
            max_val = max(volume_pivot_diff['vanilla'].max(), volume_pivot_diff['neural'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

            # Add statistics
            add_statistical_textbox(ax2, volume_pivot_diff['neural'], volume_pivot_diff['vanilla'], 'volume')

        ax2.set_xlabel('Vanilla Setup - Final Pareto Volume')
        ax2.set_ylabel('Neural Setup - Final Pareto Volume')
        ax2.set_title(f'{difficulty} Equations - Final Pareto Volume')

    plt.tight_layout()
    return fig


def create_complexity_groups(eq_df):
    """
    Create complexity and evaluation std groups from equation dataframe.

    Parameters:
    - eq_df: DataFrame with equation metadata

    Returns:
    - eq_df: DataFrame with added grouping columns
    """
    # Use log transform to reduce skewness, then quantile-based grouping
    log_complexity = np.log(eq_df['complexity_simplified'] + 1e-8)  # Add small value to avoid log(0)
    log_eval_std = np.log(eq_df['eval_std'] + 1e-8)  # Add small value to avoid log(0)

    eq_df['complexity_group'] = pd.qcut(log_complexity,
                                       q=3,
                                       labels=['Low', 'Medium', 'High'],
                                       duplicates='drop')

    eq_df['eval_std_group'] = pd.qcut(log_eval_std,
                                     q=3,
                                     labels=['Low', 'Medium', 'High'],
                                     duplicates='drop')
    return eq_df


def plot_complexity_analysis(filtered_df, eq_df):
    """
    Create complexity and evaluation std analysis plots.

    Parameters:
    - filtered_df: DataFrame with results
    - eq_df: DataFrame with equation metadata and groups

    Returns:
    - fig1, fig2: matplotlib figure objects for complexity and eval std
    """
    # Create a proper mapping from equation ID to groups
    # Assume eq_df has an 'eq' column or the index represents equation IDs
    if 'eq' in eq_df.columns:
        eq_groups = eq_df[['eq', 'complexity_group', 'eval_std_group']].copy()
    else:
        eq_groups = eq_df[['complexity_group', 'eval_std_group']].copy()
        eq_groups['eq'] = eq_groups.index

    # Merge with equation groups
    filtered_df_with_groups = filtered_df.merge(eq_groups, on='eq', how='left')

    # Plot by complexity groups
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Performance Comparison by Equation Complexity', fontsize=16)

    complexity_groups = ['Low', 'Medium', 'High']
    colors = ['green', 'orange', 'red']

    for i, complexity in enumerate(complexity_groups):
        # Filter data for this complexity level
        comp_data = filtered_df_with_groups[filtered_df_with_groups['complexity_group'] == complexity]

        # Create pivot tables for this complexity level (like difficulty analysis)
        loss_pivot_comp = comp_data.pivot_table(
            index='eq', columns='setup', values='final_min_loss', aggfunc='mean'
        ).dropna()

        volume_pivot_comp = comp_data.pivot_table(
            index='eq', columns='setup', values='final_pareto_volume', aggfunc='mean'
        ).dropna()

        # Plot 1: Final Min Loss comparison
        ax1 = axes[0, i]
        if 'neural' in loss_pivot_comp.columns and 'vanilla' in loss_pivot_comp.columns and len(loss_pivot_comp) > 0:
            ax1.scatter(loss_pivot_comp['vanilla'], loss_pivot_comp['neural'],
                       alpha=0.7, s=50, color=colors[i])

            # Add diagonal line
            min_val = min(loss_pivot_comp['vanilla'].min(), loss_pivot_comp['neural'].min())
            max_val = max(loss_pivot_comp['vanilla'].max(), loss_pivot_comp['neural'].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

            # Add statistics
            add_statistical_textbox(ax1, loss_pivot_comp['neural'], loss_pivot_comp['vanilla'], 'loss')

            # Only set log scale if we have positive data
            if min_val > 0:
                ax1.set_xscale('log')
                ax1.set_yscale('log')

        ax1.set_xlabel('Vanilla Setup - Final Min Loss')
        ax1.set_ylabel('Neural Setup - Final Min Loss')
        ax1.set_title(f'{complexity} Complexity - Final Min Loss')

        # Plot 2: Final Pareto Volume comparison
        ax2 = axes[1, i]
        if 'neural' in volume_pivot_comp.columns and 'vanilla' in volume_pivot_comp.columns and len(volume_pivot_comp) > 0:
            ax2.scatter(volume_pivot_comp['vanilla'], volume_pivot_comp['neural'],
                       alpha=0.7, s=50, color=colors[i])

            # Add diagonal line
            min_val = min(volume_pivot_comp['vanilla'].min(), volume_pivot_comp['neural'].min())
            max_val = max(volume_pivot_comp['vanilla'].max(), volume_pivot_comp['neural'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

            # Add statistics
            add_statistical_textbox(ax2, volume_pivot_comp['neural'], volume_pivot_comp['vanilla'], 'volume')

        ax2.set_xlabel('Vanilla Setup - Final Pareto Volume')
        ax2.set_ylabel('Neural Setup - Final Pareto Volume')
        ax2.set_title(f'{complexity} Complexity - Final Pareto Volume')

    plt.tight_layout()

    # Plot by evaluation standard deviation groups
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Performance Comparison by Equation Evaluation Standard Deviation', fontsize=16)

    eval_std_groups = ['Low', 'Medium', 'High']

    for i, eval_std in enumerate(eval_std_groups):
        # Filter data for this eval std level
        std_data = filtered_df_with_groups[filtered_df_with_groups['eval_std_group'] == eval_std]

        # Create pivot tables for this eval std level
        loss_pivot_std = std_data.pivot_table(
            index='eq', columns='setup', values='final_min_loss', aggfunc='mean'
        ).dropna()

        volume_pivot_std = std_data.pivot_table(
            index='eq', columns='setup', values='final_pareto_volume', aggfunc='mean'
        ).dropna()

        # Plot 1: Final Min Loss comparison
        ax1 = axes[0, i]
        if 'neural' in loss_pivot_std.columns and 'vanilla' in loss_pivot_std.columns and len(loss_pivot_std) > 0:
            ax1.scatter(loss_pivot_std['vanilla'], loss_pivot_std['neural'],
                       alpha=0.7, s=50, color=colors[i])

            # Add diagonal line
            min_val = min(loss_pivot_std['vanilla'].min(), loss_pivot_std['neural'].min())
            max_val = max(loss_pivot_std['vanilla'].max(), loss_pivot_std['neural'].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

            # Add statistics
            add_statistical_textbox(ax1, loss_pivot_std['neural'], loss_pivot_std['vanilla'], 'loss')

            # Only set log scale if we have positive data
            if min_val > 0:
                ax1.set_xscale('log')
                ax1.set_yscale('log')

        ax1.set_xlabel('Vanilla Setup - Final Min Loss')
        ax1.set_ylabel('Neural Setup - Final Min Loss')
        ax1.set_title(f'{eval_std} Eval Std - Final Min Loss')

        # Plot 2: Final Pareto Volume comparison
        ax2 = axes[1, i]
        if 'neural' in volume_pivot_std.columns and 'vanilla' in volume_pivot_std.columns and len(volume_pivot_std) > 0:
            ax2.scatter(volume_pivot_std['vanilla'], volume_pivot_std['neural'],
                       alpha=0.7, s=50, color=colors[i])

            # Add diagonal line
            min_val = min(volume_pivot_std['vanilla'].min(), volume_pivot_std['neural'].min())
            max_val = max(volume_pivot_std['vanilla'].max(), volume_pivot_std['neural'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

            # Add statistics
            add_statistical_textbox(ax2, volume_pivot_std['neural'], volume_pivot_std['vanilla'], 'volume')

        ax2.set_xlabel('Vanilla Setup - Final Pareto Volume')
        ax2.set_ylabel('Neural Setup - Final Pareto Volume')
        ax2.set_title(f'{eval_std} Eval Std - Final Pareto Volume')

    plt.tight_layout()

    return fig1, fig2


def print_summary_statistics(filtered_df, comparison_metrics=None):
    """
    Print comprehensive summary statistics for all metrics.

    Parameters:
    - filtered_df: DataFrame with results
    - comparison_metrics: List of metrics to compare (optional)
    """
    if comparison_metrics is None:
        comparison_metrics = [
            'min_loss_step_100', 'max_pareto_volume_step_100',
            'min_loss_step_600', 'max_pareto_volume_step_600',
            'min_loss_step_929', 'max_pareto_volume_step_929'
        ]

    print("=== COMPREHENSIVE METRIC COMPARISON ===")
    summary_stats = []

    for metric in comparison_metrics:
        if metric not in filtered_df.columns:
            continue

        neural_data = filtered_df[filtered_df['setup'] == 'neural'][metric]
        vanilla_data = filtered_df[filtered_df['setup'] == 'vanilla'][metric]

        # Perform statistical test
        try:
            stat, pval = stats.wilcoxon(neural_data, vanilla_data)
            test_result = f"p={pval:.4f}"
        except:
            test_result = "N/A"

        summary_stats.append({
            'Metric': metric,
            'Neural_Mean': neural_data.mean(),
            'Vanilla_Mean': vanilla_data.mean(),
            'Neural_Median': neural_data.median(),
            'Vanilla_Median': vanilla_data.median(),
            'Wilcoxon_p': test_result
        })

    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False))