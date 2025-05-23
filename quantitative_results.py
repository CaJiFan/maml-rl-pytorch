import numpy as np
import os
import pandas as pd

def analyze_maml_results(results_files):
    """
    Loads results from .npz files generated by test.py, calculates average
    validation returns, and organizes the data for comparison.

    Args:
        results_files (list): A list of dictionaries, where each dictionary
                              contains information about a results file:
                              {'filepath': 'path/to/results.npz',
                               'env': 'EnvironmentName',
                               'base_rl': 'TRPO' or 'PPO',
                               'num_tasks': 10, 200, or 500}

    Returns:
        pandas.DataFrame: A DataFrame containing the average validation return
                          for each experiment configuration.
    """
    all_results = []

    for exp_config in results_files:
        filepath = exp_config['filepath']
        env = exp_config['env']
        base_rl = exp_config['base_rl']
        num_tasks = exp_config['num_tasks']

        if not os.path.exists(filepath):
            print(f"Warning: File not found at {filepath}. Skipping.")
            continue

        try:
            # Load the data from the .npz file
            data = np.load(filepath)
            # The valid_returns array contains returns for each meta-test task
            valid_returns = data['valid_returns']

            # Calculate the average return across all meta-test tasks
            average_valid_return = np.mean(valid_returns)

            # Store the results
            all_results.append({
                'Environment': env,
                'Base RL Algorithm': base_rl,
                'Num Tasks (Outer Loop)': num_tasks,
                'Average Validation Return': average_valid_return
            })

        except Exception as e:
            print(f"Error loading or processing file {filepath}: {e}")
            continue

    # Convert the list of results into a pandas DataFrame for easy analysis
    results_df = pd.DataFrame(all_results)

    return results_df

if __name__ == '__main__':
    # --- Dynamic Results File Discovery ---
    base_results_dir = 'results' # Your main results directory

    experiment_configs = []

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
                        experiment_configs.append({
                            'filepath': results_file_path,
                            'env': env_name,
                            'base_rl': base_rl_algo.upper(), # Store as uppercase TRPO/PPO
                            'num_tasks': num_tasks_int
                        })
                    except ValueError:
                        print(f"Warning: Could not convert {num_tasks_str} to integer. Skipping.")
                else:
                    print(f"Warning: Results file not found: {results_file_path}. Skipping.")


    if not experiment_configs:
        print("No results files found matching the expected structure.")
    else:
        # Analyze the results
        analysis_df = analyze_maml_results(experiment_configs)

        # Print the results DataFrame
        print("--- Analysis Results ---")
        print(analysis_df)

        # --- How to use this for your report tables ---
        # You can now use this DataFrame to populate your quantitative results tables.
        # Filter and pivot the DataFrame as shown in the previous example
        # to generate tables for each environment, comparing TRPO and PPO
        # across the different numbers of tasks.
