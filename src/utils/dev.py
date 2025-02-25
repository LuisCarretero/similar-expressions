import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from scipy.stats import norm
from typing import Union
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig

from src.utils.parsing import logits_to_infix, eval_from_logits


def calc_properties_and_partials(y_values: torch.Tensor):
    # Mean
    mean = torch.mean(y_values, dim=1)

    # Upwardness
    eps = 1e-4
    y_diff = torch.diff(y_values, dim=1)
    is_increasing = y_diff > eps
    is_decreasing = y_diff < -eps
    change_flag = is_increasing.int() - is_decreasing.int()
    upwardness = torch.mean(change_flag.float(), dim=1)

    # Oscillation
    is_extremum = torch.abs(torch.diff(change_flag, dim=1)) > 1
    oscillations = torch.sum(is_extremum, dim=1)

    # Non-convexity ratio (NCR)
    # Doing this on every point for now
    eps = 1e-9
    kernel = torch.tensor([-0.5, 1, -0.5]).unsqueeze(0).unsqueeze(0)
    y_values_tensor = y_values.unsqueeze(1)  # Add channel dimension
    convolved = F.conv1d(y_values_tensor, kernel, padding=1)
    is_convex = convolved.squeeze(1) <= eps
    non_convexity_ratio = 1 - torch.mean(is_convex.float(), dim=1)

    return mean, upwardness, oscillations, non_convexity_ratio, is_extremum, change_flag, is_convex

def calc_properties(y_values: torch.Tensor):
    mean, upwardness, oscillations, non_convexity_ratio, _, _, _ = calc_properties_and_partials(y_values)
    return mean, upwardness, oscillations, non_convexity_ratio


def plot_property_distributions(properties):
    """
    Plot histograms of property distributions.
    
    Args:
    properties (np.ndarray): Array of shape (n_samples, 4) containing the properties.
    """
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Distribution of Properties')

    property_names = ['Mean', 'Upwardness', 'ln(Oscillations)', 'Non-convexity Ratio (NCR)']

    for i, (ax, name) in enumerate(zip(axs.flatten(), property_names)):
        ax.hist(properties[:, i], bins=50, edgecolor='black')
        ax.set_title(name)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

        mean = np.mean(properties[:, i])
        std = np.std(properties[:, i])
        ax.text(0.05, 0.95, f'Mean: {mean:.3f}\nStd: {std:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_tsne_properties(z: np.ndarray, properties: np.ndarray, title='t-SNE visualization of latent space'):
    """
    Create scatter plots for all four properties using t-SNE visualization.
    
    Args:
    z (np.array): 2D t-SNE embedding of the latent space
    properties (np.array): Array of properties for each data point
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    property_names = ['Mean', 'Upwardness', 'ln(Oscillations)', 'Non-convexity Ratio (NCR)']

    for i, (ax, property_name) in enumerate(zip(axs.flatten(), property_names)):
        scatter = ax.scatter(z[:, 0], z[:, 1], c=properties[:, i], cmap='Spectral', alpha=1, s=2)
        
        fig.colorbar(scatter, ax=ax)
        ax.set_title(f'Colored by {property_name}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_values_with_property(y_values: torch.Tensor, val_x=np.linspace(-10, 10, 100)):
    val_x = val_x.squeeze()
    assert y_values.shape[0] == 1, 'Only batch size 1 supported'
    
    mean, upwardness, oscillations, non_convexity_ratio, is_extremum, change_flag, is_convex = calc_properties_and_partials(y_values)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # First subplot: Extrema and increasing/decreasing
    extremum_idx = np.where(is_extremum[0, :])[0] + 1
    ax1.scatter(val_x[extremum_idx], y_values[0, extremum_idx], color='red', label='Extremum')

    for change, color in {-1: 'r', 1: 'g', 0: 'b'}.items():
        mask = (change_flag == change).squeeze()
        mask = np.concatenate([[False], mask, [False]])
        x = np.stack([val_x[mask[:-1]], val_x[mask[1:]]], axis=1).T 
        y = np.stack([y_values.squeeze()[mask[:-1]], y_values.squeeze()[mask[1:]]], axis=1).T
        
        ax1.plot(x, y, color=color)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.plot([], [], color='r', label='Decreasing')
    ax1.plot([], [], color='g', label='Increasing')
    ax1.plot([], [], color='b', label='No significant change')
    ax1.legend()
    ax1.set_title('Extrema and Increasing/Decreasing')

    # Add scalar values to the first subplot
    ax1.text(0.05, 0.95, f'NCR: {non_convexity_ratio[0]:.3f}', transform=ax1.transAxes, verticalalignment='top')
    ax1.text(0.05, 0.90, f'Mean: {mean[0]:.3f}', transform=ax1.transAxes, verticalalignment='top')
    ax1.text(0.05, 0.85, f'Upwardness: {upwardness[0]:.3f}', transform=ax1.transAxes, verticalalignment='top')
    ax1.text(0.05, 0.80, f'Oscillations: {oscillations[0]:.0f}', transform=ax1.transAxes, verticalalignment='top')

    # Second subplot: Local convexity
    for is_convex_current, color in {True: 'g', False: 'r'}.items():
        mask = (is_convex == is_convex_current).squeeze()
        mask = np.concatenate([[False], mask[:-1], [False]])
        x = np.stack([val_x[mask[:-1]], val_x[mask[1:]]], axis=1).T 
        y = np.stack([y_values.squeeze()[mask[:-1]], y_values.squeeze()[mask[1:]]], axis=1).T
        
        ax2.plot(x, y, color=color)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.plot([], [], color='r', label='Non-convex')
    ax2.plot([], [], color='g', label='Convex')
    ax2.legend()
    ax2.set_title('Local Convexity')

    plt.tight_layout()
    plt.show()


def plot_onehot(onehot_matrix, xticks, apply_softmax=False, figsize=(10, 5)):
    onehot_matrix = onehot_matrix.copy()
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    if apply_softmax:
        onehot_matrix[:, :-1] = F.softmax(onehot_matrix[:, :-1], axis=-1)
    im1 = ax1.imshow(onehot_matrix[:, :-1])
    im2 = ax2.imshow(np.expand_dims(onehot_matrix[:, -1], axis=1))

    ax1.set_ylabel('Sequence')
    ax1.set_xlabel('Rule')

    ax1.set_xticks(range(len(xticks)), xticks, rotation='vertical')
    ax2.set_xticks([0], ['[CON]'], rotation='vertical')
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.show()


def plot_original_vs_predicted_properties(properties, properties_pred):
    """
    Create scatter plots comparing original and predicted properties.
    
    Args:
    properties (np.array): Original property values
    properties_pred (np.array): Predicted property values
    """
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Original vs Predicted Properties', fontsize=16)

    property_names = ['Mean', 'Upwardness', 'ln(Oscillations)', 'Non-convexity Ratio (NCR)']

    for i, (ax, property_name) in enumerate(zip(axs.flatten(), property_names)):
        original = properties[:, i]
        predicted = properties_pred[:, i]
        
        ax.scatter(original, predicted, alpha=0.5)
        ax.set_xlabel('Original')
        ax.set_ylabel('Predicted')
        ax.set_title(property_name)
        
        # Add a diagonal line for reference
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        
        # Calculate and display mean and std for original and predicted
        orig_mean, orig_std = np.mean(original), np.std(original)
        pred_mean, pred_std = np.mean(predicted), np.std(predicted)
        
        ax.text(0.05, 0.95, f'Original: μ={orig_mean:.2f}, σ={orig_std:.2f}', 
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.89, f'Predicted: μ={pred_mean:.2f}, σ={pred_std:.2f}', 
                transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.show()


def plot_var_distributions(mean_of_var_train, mean_of_var_test, std_of_var_train, std_of_var_test):
    def calculate_histogram(data_train, data_test, num_bins=101):
        bins = np.linspace(min(data_train.min(), data_test.min()),
                           max(data_train.max(), data_test.max()),
                           num_bins)
        train_hist, _ = np.histogram(data_train, bins=bins, density=True)
        test_hist, _ = np.histogram(data_test, bins=bins, density=True)
        return bins, train_hist, test_hist

    bins_mean, train_mean_hist, test_mean_hist = calculate_histogram(mean_of_var_train, mean_of_var_test)
    bins_std, train_std_hist, test_std_hist = calculate_histogram(std_of_var_train, std_of_var_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    def plot_distribution(ax, bins, train_hist, test_hist, xlabel, title):
        ax.step(bins[:-1], train_hist, where='post', color='blue', alpha=0.7, label='Train')
        ax.step(bins[:-1], test_hist, where='post', color='red', alpha=0.7, label='Test')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()

    plot_distribution(ax1, bins_mean, train_mean_hist, test_mean_hist, 'Mean of Var', 'Distribution of Mean of Var (Train vs Test)')
    plot_distribution(ax2, bins_std, train_std_hist, test_std_hist, 'Std of Var', 'Distribution of Std of Var (Train vs Test)')

    plt.tight_layout()
    plt.show()


def plot_latent_distribution(z_train: np.ndarray, cfg: DictConfig):
    num_dims = cfg.model.z_size
    num_bins = 100
    
    def calculate_histogram_data(z_train):
        hist_data = []
        bin_edges = []
        for dim in range(num_dims):
            hist, bins = np.histogram(z_train[:, dim], bins=num_bins, density=True)
            hist_data.append(hist)
            bin_edges.append(bins)
        bin_centers = [(bins[:-1] + bins[1:]) / 2 for bins in bin_edges]
        normalized_hist_data = [hist / np.max(hist) for hist in hist_data]
        return bin_centers, normalized_hist_data

    def plot_distributions(ax, bin_centers, normalized_hist_data, x_range, title):
        ax.set_title(title, fontsize=16)
        for dim in range(num_dims):
            ax.plot(bin_centers[dim], normalized_hist_data[dim], linewidth=2, alpha=0.5)
        
        x = np.concatenate([np.linspace(x_range[0], -cfg.training.sampling.prior_std, 1000),
                            np.linspace(-cfg.training.sampling.prior_std, cfg.training.sampling.prior_std, 8000),
                            np.linspace(cfg.training.sampling.prior_std, x_range[1], 1000)])
        
        for label, std in [('Sampling Gaussian', cfg.training.sampling.eps), 
                           (f'Prior ($\\sigma$={cfg.training.sampling.prior_std:.2f})', cfg.training.sampling.prior_std)]:
            gaussian = norm.pdf(x, 0, std)
            normalized_gaussian = gaussian / np.max(gaussian)
            ax.plot(x, normalized_gaussian, 'k', 
                    linestyle='-' if 'Sampling' in label else '--', 
                    linewidth=2, label=label)
        
        ax.set_xlim(x_range)
        ax.set_xlabel('Latent Dimension Value')
        ax.set_ylabel('Normalized Density')
        ax.legend()

    bin_centers, normalized_hist_data = calculate_histogram_data(z_train)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    plot_distributions(ax1, bin_centers, normalized_hist_data, 
                       (-7*cfg.training.sampling.prior_std, 7*cfg.training.sampling.prior_std),
                       'Distribution of Latent Dimensions')
    
    plot_distributions(ax2, bin_centers, normalized_hist_data, 
                       np.array([-1, 1])*cfg.training.sampling.eps*40,
                       'Zoomed Distribution')

    plt.tight_layout()
    plt.show()


def plot_latent_distances(z_train: np.ndarray, z_test: np.ndarray, title: str, sample_size: int = 1000):

    # Calculate norms and pairwise distances for train data
    train_norms = np.linalg.norm(z_train, axis=1)
    sampled_train = z_train[np.random.choice(z_train.shape[0], sample_size, replace=False)]
    pairwise_distances_train = np.linalg.norm(sampled_train[:, np.newaxis] - sampled_train, axis=2)
    l2_distances_train = pairwise_distances_train[np.triu_indices(sample_size, k=1)]

    # Calculate norms and pairwise distances for test data
    test_norms = np.linalg.norm(z_test, axis=1)
    sampled_test = z_test[np.random.choice(z_test.shape[0], sample_size, replace=False)]
    pairwise_distances_test = np.linalg.norm(sampled_test[:, np.newaxis] - sampled_test, axis=2)
    l2_distances_test = pairwise_distances_test[np.triu_indices(sample_size, k=1)]

    # Plot histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot histogram of norms
    norm_bins = np.linspace(min(train_norms.min(), test_norms.min()),
                            max(train_norms.max(), test_norms.max()),
                            50)
    ax1.hist(train_norms, bins=norm_bins, alpha=0.5, label='Train', edgecolor='black', density=True)
    ax1.hist(test_norms, bins=norm_bins, alpha=0.5, label='Test', edgecolor='black', density=True)
    ax1.set_title('Histogram of Latent Space Vector Norms')
    ax1.set_xlabel('Norm')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot histogram of L2 distances
    l2_bins = np.linspace(min(l2_distances_train.min(), l2_distances_test.min()),
                          max(l2_distances_train.max(), l2_distances_test.max()),
                          50)
    ax2.hist(l2_distances_train, bins=l2_bins, alpha=0.5, label='Train', edgecolor='black', density=True)
    ax2.hist(l2_distances_test, bins=l2_bins, alpha=0.5, label='Test', edgecolor='black', density=True)
    ax2.set_title('Histogram of Pairwise L2 Distances')
    ax2.set_xlabel('L2 Distance')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def slerp(v0: np.ndarray, v1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation. FIXME: Vectorize."""
    # Compute the cosine of the angle between the two vectors
    omega = np.arccos(np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)))
    sin_omega = np.sin(omega)

    s0 = np.sin((1 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega
    return s0[:, np.newaxis] * v0 + s1[:, np.newaxis] * v1

def plot_sample_distribution(val_x: Union[torch.Tensor, np.ndarray], values_true: torch.Tensor = None, values_pred: torch.Tensor = None, values_neigh: torch.Tensor = None, plot_extrema: bool = True, ax = None):
    
    assert any([x is not None for x in [values_true, values_pred, values_neigh]]), 'At least one of values_true, values_pred, values_neigh must be provided'

    val_x = val_x.squeeze()
    if isinstance(val_x, torch.Tensor):
        val_x = val_x.detach().numpy()

    if values_neigh is not None:
        values_neigh = values_neigh.detach().numpy()
        mean_samples = np.mean(values_neigh, axis=0)
        std_samples = np.std(values_neigh, axis=0)
        if plot_extrema:    
            min_samples = np.min(values_neigh, axis=0)
            max_samples = np.max(values_neigh, axis=0)

    if ax is None:
        _, ax = plt.subplots()

    if values_true is not None:
        ax.plot(val_x, values_true.squeeze().detach().numpy(), label='original', color='blue')
    if values_pred is not None:
        ax.plot(val_x, values_pred.squeeze().detach().numpy(), label='pred', color='green')
    if values_neigh is not None:
        ax.plot(val_x, mean_samples, label='sample mean', color='red')
        ax.fill_between(val_x, 
                        mean_samples - std_samples, 
                        mean_samples + std_samples, 
                        alpha=0.3, color='red', label='sample std')

        if plot_extrema:
            ax.plot(val_x, min_samples, label='sample min', color='orange', linestyle='--')
            ax.plot(val_x, max_samples, label='sample max', color='purple', linestyle='--')

    ax.set_ylim(-10, 10)


def calc_and_plot_samples(model, x: torch.Tensor, values_true: torch.Tensor, n_samples: int, ax = None, mode='value', val_x=None, value_transform=None, var_multiplier=1, use_const_var=False):
    mean, ln_var = model.encoder(x)
    ln_var = ln_var + torch.log(torch.ones_like(ln_var) * var_multiplier)
    if use_const_var:
        assert isinstance(use_const_var, Union[float, int]), 'use_const_var must be a float/int or tensor if use_const_var is not False'
        ln_var = torch.log(torch.ones_like(ln_var) * use_const_var)

    if mode == 'value':
        values_pred = model.value_decoder(mean)
        z = model.sample(mean.repeat(n_samples, 1), ln_var.repeat(n_samples, 1))
        values_neigh = model.value_decoder(z)

    elif mode == 'syntax':
        assert val_x is not None, 'val_x must be provided for syntax mode'
        assert value_transform is not None, 'value_transform must be provided for syntax mode'

        logits_pred = model.decoder(mean)
        res_raw = eval_from_logits(logits_pred.squeeze(), val_x.squeeze())
        try:
            res = res_raw.astype(np.float32)
        except TypeError:
            res = np.zeros_like(res_raw, dtype=np.float32)
        values_pred = value_transform(torch.tensor(res).unsqueeze(0)).squeeze()

        # Sample and decode from neighbourhood
        z = model.sample(mean.repeat(n_samples, 1), ln_var.repeat(n_samples, 1))
        logits_neigh = model.decoder(z)
        
        values_neigh = torch.empty([n_samples, len(val_x)])
        
        for i in tqdm(range(n_samples)):
            try:
                res_raw = eval_from_logits(logits_neigh[i, ...], val_x.squeeze())
                res = res_raw.astype(np.float32)
            except (TypeError, AssertionError):
                res = np.zeros_like(res_raw, dtype=np.float32)
                # print(f'Warning: Failed to decode logits {i}')

            values_neigh[i, ...] = value_transform(torch.tensor(res).unsqueeze(0)).squeeze()
        
    plot_sample_distribution(val_x, values_true, values_pred, values_neigh, ax=ax)


def calc_and_plot_samples_grid(model, x: torch.Tensor, values_true: torch.Tensor, n_samples: int = 100, idx = None, mode='value', val_x=None, value_transform=None, var_multiplier=1, use_const_var=False):
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))

    for i in range(16):
        ax = axes[i//4, i%4]
        calc_and_plot_samples(model, x[i].unsqueeze(0), values_true[i], n_samples, ax, mode=mode, val_x=val_x, value_transform=value_transform, var_multiplier=var_multiplier, use_const_var=use_const_var)
        
        if idx is not None:
            ax.set_title(f'Example {idx[i]}', fontsize=10)

        if i == 0:
            ax.legend(fontsize=8)
        # Remove x ticks for all but the bottom row
        if i // 4 < 3:  # If not in the bottom row
            ax.set_xticks([])

    plt.tight_layout()
    plt.show()

def plot_interpolation(model, val_x: torch.Tensor, z_start: np.ndarray, z_end: np.ndarray, start_true: np.ndarray, end_true: np.ndarray, value_transform, num_steps: int = 20, interp_mode='slerp'):
    assert num_steps > 3, 'num_steps must be greater than 3.'
    print(f'Distance: {np.linalg.norm(z_end - z_start)}')

    # Interpolate in latent space
    alpha = np.linspace(0, 1, num_steps)
    if interp_mode == 'slerp':
        z_interp = slerp(z_start, z_end, alpha)
    elif interp_mode == 'linear':
        z_interp = z_start[np.newaxis, ...] * (1 - alpha[:, np.newaxis]) + z_end[np.newaxis, ...] * alpha[:, np.newaxis]
    else:
        raise ValueError(f'Interpolation mode {interp_mode} not supported')
    
    # Decode into values and logits
    values_interp = model.value_decoder(torch.tensor(z_interp.astype(np.float32)))
    logits_interp = model.decoder(torch.tensor(z_interp.astype(np.float32)))

    # Decode logits into values
    values_interp_syntax = torch.empty_like(values_interp)
    for idx in range(0, logits_interp.shape[0]):
        res = eval_from_logits(logits_interp[idx, ...], val_x.squeeze())
        try:
            res = res.astype(np.float32)
            values_interp_syntax[idx, ...] = value_transform(torch.tensor(res).unsqueeze(0)).squeeze()
        except TypeError:
            print(f'Warning: Failed to decode logits {idx}')
            values_interp_syntax[idx, ...] = torch.zeros_like(res, dtype=torch.float32)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_value_interpolation(ax1, val_x, values_interp, start_true, end_true, 'Value Decoding')
    plot_value_interpolation(ax2, val_x, values_interp_syntax, start_true, end_true, 'Syntax Decoding')
    add_colorbar(fig, ax2)
    plt.tight_layout()
    plt.show()

def plot_value_interpolation(ax: plt.Axes, val_x: torch.Tensor, values_interp: torch.Tensor, start_true: np.ndarray, end_true: np.ndarray, title: str):

    if values_interp.isnan().any():
        print('Warning: values_interp contains NaNs')

    # Pred values (interpolated)
    cmap = plt.get_cmap('rainbow')
    for idx, value in enumerate(values_interp[1:-1]):
        color = cmap(idx / (len(values_interp) - 3))
        ax.plot(val_x.squeeze(), value.detach().numpy(), color=color, alpha=0.5)

    # Pred values (ends)
    ax.plot(val_x.squeeze(), values_interp[0].squeeze(), label='start (pred)', color='blue', linewidth=2)
    ax.plot(val_x.squeeze(), values_interp[-1].squeeze(), label='end (pred)', color='red', linewidth=2)

    # True values (ends)
    ax.plot(val_x.squeeze(), start_true, label='start (true)', color='lightblue', linewidth=2, linestyle='--')
    ax.plot(val_x.squeeze(), end_true, label='end (true)', color='pink', linewidth=2, linestyle='--')

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

def add_colorbar(fig, ax):
    cmap = plt.get_cmap('rainbow')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Interpolation Progress')
