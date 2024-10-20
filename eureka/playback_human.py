import os
import re
import subprocess
import yaml

# Base directory where all simulation runs are stored
base_dir = "/home/vandriel/Documents/GitHub/Eureka/isaacgymenvs/isaacgymenvs/outputs/train"

# Function to extract the task name from env.py
def get_task_name(env_file_path):
    print(f"Reading task name from: {env_file_path}")
    with open(env_file_path, 'r') as file:
        for line in file:
            match = re.match(r'class\s+(\w+)\(VecTask\):', line)
            if match:
                task_name = match.group(1)
                print(f"Found task name: {task_name}")
                return task_name
    print(f"No task name found in {env_file_path}")
    return None

# Function to extract experiment name and gravity from config.yaml
def get_experiment_config(config_file_path):
    print(f"Reading experiment config from: {config_file_path}")
    with open(config_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        experiment_name = yaml_data.get('experiment', None)
        gravity = yaml_data.get('task', {}).get('sim', {}).get('gravity', None)
        
        if experiment_name:
            print(f"Found experiment name: {experiment_name}")
        if gravity:
            print(f"Found gravity: {gravity}")
        else:
            print(f"No gravity found in {config_file_path}")
            
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

# Collect and sort available simulation directories by date-time format (oldest first)
tasks = []
for subdir in sorted(os.listdir(base_dir)):
    subdir_path = os.path.join(base_dir, subdir)
    print(f"\nChecking directory: {subdir_path}")
    if os.path.isdir(subdir_path):
        env_file = os.path.join(subdir_path, "env.py")
        if os.path.isfile(env_file):
            print(f"env.py found in {subdir_path}")
            task_name = get_task_name(env_file)
            if task_name:
                # Search for config.yaml in the 'runs' subdirectories (since experiment_name is not yet known)
                runs_dir = os.path.join(subdir_path, "runs")
                if os.path.isdir(runs_dir):
                    for run_subdir in os.listdir(runs_dir):
                        config_file_path = os.path.join(runs_dir, run_subdir, "config.yaml")
                        print(f"Looking for config.yaml at: {config_file_path}")
                        if os.path.isfile(config_file_path):
                            experiment_name, gravity = get_experiment_config(config_file_path)  # Fetch experiment and gravity
                            if experiment_name:
                                # Search for the correct .pth file in the runs directory
                                pth_file = find_pth_file(runs_dir, experiment_name)
                                if pth_file:
                                    tasks.append((task_name, experiment_name, subdir, gravity))  # Include gravity
                                else:
                                    print(f"No .pth file found for {experiment_name} in {runs_dir}")
                        else:
                            print(f"No config.yaml found in {config_file_path}")
                else:
                    print(f"No 'runs' directory found in {subdir_path}")
        else:
            print(f"No env.py found in {subdir_path}")
    else:
        print(f"{subdir_path} is not a directory")

# Display tasks for selection
if not tasks:
    print("\nNo valid simulations with .pth files found.")
else:
    print("\nAvailable simulations (sorted by date, oldest first):")
    for idx, (task, experiment, date, gravity) in enumerate(tasks, 1):
        print(f"{idx}. Task: {task}, Experiment: {experiment}, Date: {date}, Gravity: {gravity}")

    # Prompt user to select a simulation
    selection = int(input("Enter the number of the simulation to test: ")) - 1
    selected_task, selected_experiment, selected_date, selected_gravity = tasks[selection]

    # Format gravity properly for command line use
    gravity_str = "[" + ", ".join(map(str, selected_gravity)) + "]"

    # Re-construct the correct checkpoint path using the found .pth file
    runs_dir = os.path.join(base_dir, selected_date, "runs")
    checkpoint_path = find_pth_file(runs_dir, selected_experiment)
    if checkpoint_path:
        print(f"\nRunning test with checkpoint at: {checkpoint_path}")
        
        command = [
            "python", "/home/vandriel/Documents/GitHub/Eureka/isaacgymenvs/isaacgymenvs/train.py",
            f"task={selected_task}",
            f"checkpoint={checkpoint_path}",
            f"+task.sim.gravity={gravity_str}",  # Pass formatted gravity here
            "test=True", "num_envs=4096", "headless=True", "force_render=False"
        ]

        # Execute the command
        print(f"Running test for {selected_task} (Experiment: {selected_experiment}, Gravity: {gravity_str}) from {selected_date}...")
        subprocess.run(command)
    else:
        print(f"Failed to find a valid checkpoint path for {selected_experiment}")
