import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn often makes plots look nicer

# Set a style for the plots
sns.set_theme(style="whitegrid")

def load_and_process_results(base_results_dir='results'):
    """
    Loads results from .npz files based on the directory structure,
    calculates mean and standard deviation of validation returns for each run.

    Args:
        base_results_dir (str): The base directory containing the results
                                (e.g., 'results/ppo/...', 'results/trpo/...').

    Returns:
        pandas.DataFrame: A DataFrame containing the processed results
                          with columns for Environment, Base RL Algorithm,
                          Num Tasks (Outer Loop), Mean Return, and Std Dev Return.
    """
    all_processed_results = []

    # Iterate through base RL algorithms (ppo, trpo)
    for base_rl_algo in ['ppo', 'trpo']:
        base_rl_dir = os.path.join(base_results_dir, base_rl_algo)

        if not os.path.isdir(base_rl_dir):
            print(f"Warning: Directory not found: {base_rl_dir}. Skipping.")
            continue

        # Iterate through environments
        for env_name in os.listdir(base_rl_dir):
            env_dir = os.path.join(base_rl_dir, env_name)

            if not os.path.isdir(env_dir):
                continue # Skip if not a directory

            # Iterate through number of tasks (10, 200, 500)
            for num_tasks_str in ['10', '200', '500']:
                num_tasks_dir = os.path.join(env_dir, num_tasks_str)
                results_file_path = os.path.join(num_tasks_dir, 'results.npz')

                # Check if the results file exists
                if os.path.exists(results_file_path):
                    try:
                        num_tasks_int = int(num_tasks_str)
                        # Load the data from the .npz file
                        data = np.load(results_file_path)
                        # The valid_returns array contains returns for each meta-test task
                        valid_returns = data['valid_returns']

                        # Calculate mean and standard deviation across meta-test tasks
                        mean_return = np.mean(valid_returns)
                        std_dev_return = np.std(valid_returns)

                        # Store the processed results
                        all_processed_results.append({
                            'Environment': env_name,
                            'Base RL Algorithm': base_rl_algo.upper(), # Store as uppercase TRPO/PPO
                            'Num Tasks (Outer Loop)': num_tasks_int,
                            'Mean Return': mean_return,
                            'Std Dev Return': std_dev_return
                        })

                    except ValueError:
                        print(f"Warning: Could not convert {num_tasks_str} to integer for {results_file_path}. Skipping.")
                    except Exception as e:
                        print(f"Error loading or processing file {results_file_path}: {e}")
                        continue
                else:
                    # print(f"Warning: Results file not found: {results_file_path}. Skipping.") # Optional: uncomment to see missing files
                    pass # Silently skip if file not found, as some combinations might not exist

    # Convert the list of processed results into a pandas DataFrame
    processed_df = pd.DataFrame(all_processed_results)

    return processed_df

def plot_results(df):
    """
    Generates plots for each environment comparing TRPO vs PPO and
    different numbers of outer loop tasks.

    Args:
        df (pandas.DataFrame): DataFrame containing processed results
                               from load_and_process_results.
    """
    if df.empty:
        print("No data to plot. Please check your results directory and file paths.")
        return

    unique_environments = df['Environment'].unique()

    for env in unique_environments:
        env_df = df[df['Environment'] == env].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Ensure Num Tasks is treated as a number for sorting
        env_df['Num Tasks (Outer Loop)'] = pd.to_numeric(env_df['Num Tasks (Outer Loop)'])
        env_df = env_df.sort_values(by='Num Tasks (Outer Loop)')

        plt.figure(figsize=(10, 6))

        # Plot for each base RL algorithm
        for base_rl in ['TRPO', 'PPO']:
            algo_df = env_df[env_df['Base RL Algorithm'] == base_rl]

            if not algo_df.empty:
                # Plot the mean return as a line
                plt.plot(algo_df['Num Tasks (Outer Loop)'], algo_df['Mean Return'], marker='o', label=base_rl)

                # Add shaded region for standard deviation
                # Calculate upper and lower bounds for the shaded area
                upper_bound = algo_df['Mean Return'] + algo_df['Std Dev Return']
                lower_bound = algo_df['Mean Return'] - algo_df['Std Dev Return']

                plt.fill_between(algo_df['Num Tasks (Outer Loop)'], lower_bound, upper_bound, alpha=0.2)

        plt.title(f'Average Validation Return vs. Number of Outer Loop Tasks ({env})')
        plt.xlabel('Number of Outer Loop Tasks')
        plt.ylabel('Average Validation Return')
        plt.xticks(env_df['Num Tasks (Outer Loop)'].unique()) # Ensure x-ticks are at the task numbers
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # Save the plot
        plot_filename = f'{env}_maml_performance.png'
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")

        # plt.show() # Uncomment this line to display the plots when the script runs

if __name__ == '__main__':
    # --- Main Execution ---
    # Load and process the results from the specified directory structure
    results_dataframe = load_and_process_results('results')

    # Generate and save the plots
    plot_results(results_dataframe)

    print("\nAnalysis and plotting complete.")
    print("Generated plots are saved as PNG files in the current directory.")
