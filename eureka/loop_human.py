import os
import re
import subprocess
import yaml
import csv
import numpy as np  # For computing mean, stdev

# Base directory where all simulation runs are stored
log_dir = "/home/vandriel/Documents/GitHub/Eureka/isaacgymenvs/isaacgymenvs"
base_dir = "/home/vandriel/Documents/GitHub/Eureka/isaacgymenvs/isaacgymenvs/outputs/train"

# CSV output file path (you can change to an absolute path if needed)
csv_file_path = "simulation_results.csv"

# Function to extract the task name from env.py
def get_task_name(env_file_path):
    with open(env_file_path, 'r') as file:
        for line in file:
            match = re.match(r'class\s+(\w+)\(VecTask\):', line)
            if match:
                return match.group(1)
    return None

# Function to extract experiment name and gravity from config.yaml
def get_experiment_config(config_file_path):
    with open(config_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        experiment_name = yaml_data.get('experiment', None)
        gravity = yaml_data.get('task', {}).get('sim', {}).get('gravity', None)
        return experiment_name, gravity

# Function to find the correct .pth file if the timestamp is slightly off
def find_pth_file(runs_dir, experiment_name):
    print(f"Searching for .pth file in {runs_dir} for experiment {experiment_name}")
    for run_subdir in os.listdir(runs_dir):
        nn_dir = os.path.join(runs_dir, run_subdir, "nn")
        pth_file = os.path.join(nn_dir, f"{experiment_name}.pth")
        if os.path.isfile(pth_file):
            print(f"Found .pth file: {pth_file}")
            return pth_file
    print(f"No .pth file found for {experiment_name} in {runs_dir}")
    return None

# Ensure the file is created at the start and print where it's being written
print(f"Creating CSV file at: {csv_file_path}")
with open(csv_file_path, mode='w', newline='') as csv_file:
    fieldnames = ['Number', 'Task', 'Experiment', 'Gravity', 'Mean Consecutive Success', 'Stdev Consecutive Success', 'Max Consecutive Success']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
print(f"CSV file created successfully: {csv_file_path}")

# Collect and sort available simulation directories by date-time format (oldest first)
tasks = []
for subdir in sorted(os.listdir(base_dir)):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        env_file = os.path.join(subdir_path, "env.py")
        if os.path.isfile(env_file):
            task_name = get_task_name(env_file)
            if task_name:
                # Search for config.yaml in the 'runs' subdirectories
                runs_dir = os.path.join(subdir_path, "runs")
                if os.path.isdir(runs_dir):
                    for run_subdir in os.listdir(runs_dir):
                        config_file_path = os.path.join(runs_dir, run_subdir, "config.yaml")
                        if os.path.isfile(config_file_path):
                            experiment_name, gravity = get_experiment_config(config_file_path)
                            if experiment_name:
                                # Use the new function to find the .pth file
                                checkpoint_path = find_pth_file(runs_dir, experiment_name)
                                if checkpoint_path:
                                    tasks.append((task_name, experiment_name, subdir, gravity))

# Iterate through all tasks and run simulations
for idx, (task_name, experiment_name, subdir, gravity) in enumerate(tasks, 1):
    gravity_str = "[" + ", ".join(map(str, gravity)) + "]"
    runs_dir = os.path.join(base_dir, subdir, "runs")
    checkpoint_path = find_pth_file(runs_dir, experiment_name)
    n_test=1
    if checkpoint_path:
        # Debug: Start of each simulation
        for run_number in range(0, n_test):  # Run each simulation n_test times
            print(f"Starting simulation {idx}: Task={task_name}, Experiment={experiment_name}, Gravity={gravity_str}")
            
            # Construct the command
            command = [
                "python", "/home/vandriel/Documents/GitHub/Eureka/isaacgymenvs/isaacgymenvs/train.py",
                f"task={task_name}",
                f"checkpoint={checkpoint_path}",
                f"+task.sim.gravity={gravity_str}",
                "test=True", "num_envs=4096", "headless=True", "force_render=False"
            ]

            # Run the simulation
            subprocess.run(command)

            # Read the consecutive success statistics from the log file
            log_file_path = f"{log_dir}/consecutive_successes_log.txt"
            try:
                with open(log_file_path, "r") as log_file:
                    successes = [float(line.strip()) for line in log_file]
                    if successes:
                        slice_index = int(len(successes) * 0.9)
                        mean_success = round(np.mean(successes[slice_index:]), 3)
                        stdev_success = round(np.std(successes[slice_index:]), 3)
                        max_success = round(max(successes[slice_index:]), 3)
                    else:
                        mean_success = stdev_success = max_success = "N/A"
            except FileNotFoundError:
                mean_success = stdev_success = max_success = "N/A"

            # Write the results to the CSV file
            with open(csv_file_path, mode='a', newline='') as csv_file:  # Open in append mode to add each simulation result
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({
                    'Number': idx,
                    'Task': task_name,
                    'Experiment': experiment_name,
                    'Gravity': gravity_str,
                    'Mean Consecutive Success': mean_success,
                    'Stdev Consecutive Success': stdev_success,
                    'Max Consecutive Success': max_success
                })

            # Debug: After each simulation, confirm result is written
            print(f"Results for {task_name} - {experiment_name} written to CSV.")
