from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from collections import Counter
import random

def plot_value_distributions(val: np.ndarray, val_transformed: np.ndarray):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Original data (log scale)
    a = pd.Series(val.flatten())
    ax1.hist(a.abs(), bins=100)
    ax1.set_title('Original Value Distribution (Log Scale)')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Absolute value')

    # Transformed data (log scale)
    b = pd.Series(val_transformed.flatten())
    ax2.hist(b, bins=100)
    ax2.set_title('Transformed Value Distribution (Log Scale)')
    ax2.set_yscale('log')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Value')

    # Original data (normal scale)
    ax3.hist(a.abs(), bins=100)
    ax3.set_title('Original Value Distribution (Normal Scale)')
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Absolute value')

    # Transformed data (normal scale)
    ax4.hist(b, bins=100)
    ax4.set_title('Transformed Value Distribution (Normal Scale)')
    ax4.set_ylabel('Count')
    ax4.set_xlabel('Value')

    # Add describe() data to the plot
    desc_orig = a.describe().round(2)
    desc_trans = b.describe().round(2)

    ax1.text(0.95, 0.95, f"Stats:\n{desc_orig.to_string()}", 
             transform=ax1.transAxes, va='top', ha='right', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
             fontsize=8)

    ax2.text(0.95, 0.95, f"Stats:\n{desc_trans.to_string()}", 
             transform=ax2.transAxes, va='top', ha='right', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
             fontsize=8)

    plt.tight_layout()
    plt.show()


def detect_outliers(data, method='zscore', threshold=3):
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        return np.sum(z_scores > threshold)
    elif method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        return np.sum((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))

def analyze_outliers(data, plot_sample_size=10000):
    data = data.flatten()
    sampled_indices = np.random.choice(data.shape[0], size=plot_sample_size, replace=False)
    plot_data = data[sampled_indices]

    # Create Q-Q plot
    plt.figure(figsize=(6, 4))
    stats.probplot(plot_data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.show()

    # Detect outliers using z-score method
    z_score_outliers = detect_outliers(data, method='zscore', threshold=3)
    print(f"Number of outliers detected using z-score method: {z_score_outliers}")

    # Detect outliers using IQR method
    iqr_outliers = detect_outliers(data, method='iqr')
    print(f"Number of outliers detected using IQR method: {iqr_outliers}")

def analyze_sequences(syntax, categories):
    def seq_to_string(seq):
        return ' '.join([categories[np.argmax(token)] for token in seq if np.any(token)])

    # Convert all sequences to strings
    all_sequences = [seq_to_string(seq) for seq in syntax]

    # Create a large figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('Sequence Analysis', fontsize=20)

    # 1. Sequence length distribution
    seq_lengths = np.sum(syntax[:, :, :-1].sum(axis=2) != 0, axis=1)  # Except END token
    sns.histplot(seq_lengths, bins=range(1, max(seq_lengths)+2), discrete=True, ax=axs[0, 0])
    axs[0, 0].set_title('Sequence Length Distribution')
    axs[0, 0].set_xlabel('Sequence Length')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_xticks(range(1, max(seq_lengths)+1, 2))

    # 2. Sequence frequency distribution
    seq_counter = Counter(all_sequences)
    unique_seq_count = len(set(all_sequences))
    seq_frequencies = list(seq_counter.values())

    axs[0, 1].hist(seq_frequencies, bins=50, edgecolor='black')
    axs[0, 1].set_title('Sequence Frequency Distribution')
    axs[0, 1].set_xlabel('Frequency')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].set_yscale('log')

    textstr = f"Unique sequences: {unique_seq_count}\n\nTop 10 Most Common:\n"
    for seq, count in seq_counter.most_common(10):
        textstr += f"{seq}: {count}\n"

    axs[0, 1].text(0.95, 0.95, textstr, transform=axs[0, 1].transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Most common starting tokens
    start_tokens = [seq.split()[0] for seq in all_sequences]
    sns.countplot(y=start_tokens, order=sorted(set(start_tokens), key=start_tokens.count, reverse=True), ax=axs[1, 0])
    axs[1, 0].set_title('Most Common Starting Tokens')
    axs[1, 0].set_xlabel('Count')
    axs[1, 0].set_ylabel('Token')

    # 4. Most common ending tokens
    end_tokens = [seq.split()[-1] for seq in all_sequences]
    sns.countplot(y=end_tokens, order=sorted(set(end_tokens), key=end_tokens.count, reverse=True), ax=axs[1, 1])
    axs[1, 1].set_title('Most Common Ending Tokens')
    axs[1, 1].set_xlabel('Count')
    axs[1, 1].set_ylabel('Token')

    # 5. Sequence complexity (unique tokens per sequence)
    seq_complexity = [len(set(seq.split())) for seq in all_sequences]
    sns.histplot(seq_complexity, bins=range(1, max(seq_complexity)+2), ax=axs[2, 0])
    axs[2, 0].set_title('Sequence Complexity Distribution')
    axs[2, 0].set_xlabel('Number of Unique Tokens in Sequence')
    axs[2, 0].set_ylabel('Count')
    axs[2, 0].set_xticks(range(1, max(seq_complexity)+1, 2))

    # 6. Pairwise sequence similarity
    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1.ljust(max(len(s1), len(s2))), s2.ljust(max(len(s1), len(s2)))))

    def compute_pairwise_similarity(sequences, sample_size=1000):
        sampled_sequences = random.sample(sequences, min(len(sequences), sample_size))
        similarities = [1 - hamming_distance(s1, s2) / max(len(s1), len(s2))
                        for i, s1 in enumerate(sampled_sequences)
                        for s2 in sampled_sequences[i+1:]]
        return similarities

    similarities = compute_pairwise_similarity(all_sequences)
    sns.histplot(similarities, bins=20, ax=axs[2, 1])
    axs[2, 1].set_title('Pairwise Sequence Similarity Distribution')
    axs[2, 1].set_xlabel('Similarity (0-1)')
    axs[2, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

def analyze_syntax_tokens(syntax, categories):
    """
    Analyze and visualize the structure of syntax data.
    
    Args:
    syntax (np.array): Array of shape (n_samples, n_positions, n_categories)
    categories (list): List of category names
    
    Returns:
    None (displays plots)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axs = plt.subplots(2, 2, figsize=(25, 24))

    # 1. Frequency of each category at each position
    category_freq = syntax.sum(axis=0)
    sns.heatmap(category_freq.T, cmap='YlOrRd', ax=axs[0, 0], cbar_kws={'label': 'Frequency'}, annot=True, fmt='g')
    axs[0, 0].set_title('Category Frequency at Each Position', fontsize=16)
    axs[0, 0].set_xlabel('Position', fontsize=12)
    axs[0, 0].set_ylabel('Category', fontsize=12)
    axs[0, 0].set_xticks(range(syntax.shape[1]))
    axs[0, 0].set_xticklabels(range(1, syntax.shape[1] + 1))
    axs[0, 0].set_yticklabels(categories)

    # 2. Transition probabilities
    def compute_transition_probs(syntax):
        transitions = np.zeros((syntax.shape[2], syntax.shape[2]))
        for seq in syntax:
            for i in range(len(seq) - 1):
                from_cat = np.argmax(seq[i])
                to_cat = np.argmax(seq[i+1])
                transitions[from_cat, to_cat] += 1
        return transitions / np.sum(transitions, axis=1, keepdims=True)

    trans_probs = compute_transition_probs(syntax)
    sns.heatmap(trans_probs, cmap='coolwarm', ax=axs[0, 1], cbar_kws={'label': 'Probability'}, annot=True, fmt='.2f')
    axs[0, 1].set_title('Transition Probabilities', fontsize=16)
    axs[0, 1].set_xlabel('To Category', fontsize=12)
    axs[0, 1].set_ylabel('From Category', fontsize=12)
    axs[0, 1].set_xticklabels(categories, rotation=45, ha='right')
    axs[0, 1].set_yticklabels(categories, rotation=0)

    # 3. Aggregated category count (independent of position)
    aggregated_count = syntax.sum(axis=(0, 1))
    sns.barplot(x=categories, y=aggregated_count, ax=axs[1, 0])
    axs[1, 0].set_title('Aggregated Category Count', fontsize=16)
    axs[1, 0].set_xlabel('Category', fontsize=12)
    axs[1, 0].set_ylabel('Count', fontsize=12)
    axs[1, 0].tick_params(axis='x', rotation=45)

    # Hide the unused subplot
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_values(val, val_transformed, idx):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Original data
    ax1.set_title('First 10 Graphs: Original Values')
    ax1.plot(val_x.flatten(), val[:, idx])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Transformed data
    ax2.set_title('First 10 Graphs: Transformed Values')
    ax2.plot(val_x.flatten(), val_transformed[:, idx])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.tight_layout()
    plt.show()