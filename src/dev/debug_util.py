import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F


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
