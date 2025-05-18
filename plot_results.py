import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(values, window=5):
    """Moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')

def plot_maml_results(path_to_npz, smooth_window=5):
    # Load the saved results
    results = np.load(path_to_npz)

    train_returns = results['train_returns']
    valid_returns = results['valid_returns']

    meta_batch_size = 40

    # Create batch indices
    num_batches = len(valid_returns) // meta_batch_size 
    batch_indices = np.arange(len(valid_returns)) // meta_batch_size

    # Compute average return per batch
    train_mean = np.array([train_returns[i * meta_batch_size:(i + 1) * meta_batch_size].mean() for i in range(num_batches)])
    valid_mean = np.array([valid_returns[i * meta_batch_size:(i + 1) * meta_batch_size].mean() for i in range(num_batches)])

    # Smooth the curves
    train_smoothed = smooth_curve(train_mean, window=smooth_window)
    valid_smoothed = smooth_curve(valid_mean, window=smooth_window)

    # Plotting
    plt.figure(figsize=(10, 6))
    # plt.plot(train_smoothed, label='Train Return (Smoothed)')
    plt.plot(valid_smoothed, label='Validation Return (Smoothed)')
    plt.title('MAML Meta-Test Performance')
    plt.xlabel('Meta-Update Step (Batch)')
    plt.ylabel('Average Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, required=True, help='Path to results .npz file')
    parser.add_argument('--window', type=int, default=5, help='Smoothing window size')
    args = parser.parse_args()

    plot_maml_results(args.npz, smooth_window=args.window)
