import os
import re
import subprocess
import yaml  # To read the YAML config file

# Base directory where all simulation ROOT directories are stored
base_dir = "/home/vandriel/Documents/GitHub/Eureka/eureka/outputs/eureka"

# Function to extract task, LLM, and BESTFILE from eureka.log
def parse_log_file(log_file_path):
    task = None
    llm = None
    bestfile = None
    print(f"Reading eureka.log from: {log_file_path}")
    
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

        # Extract task and LLM from the first few lines
        for line in lines[:10]:  # Adjust if necessary based on actual log length
            if "Task:" in line:
                task_match = re.search(r'Task: (\w+)', line)
                if task_match:
                    task = task_match.group(1)
                    print(f"Found task: {task}")
            if "Using LLM:" in line:
                llm_match = re.search(r'Using LLM: (\w+)', line)
                if llm_match:
                    llm = llm_match.group(1)
                    print(f"Found LLM: {llm}")

        # Extract BESTFILE from the last few lines
        for line in lines[-10:]:  # Adjust if necessary based on actual log length
            bestfile_match = re.search(r'Best Reward Code Path: (\S+)', line)
            if bestfile_match:
                bestfile = bestfile_match.group(1)
                print(f"Found BESTFILE: {bestfile}")
                break

    if not task:
        print(f"No task found in {log_file_path}")
    if not llm:
        print(f"No LLM found in {log_file_path}")
    if not bestfile:
        print(f"No BESTFILE found in {log_file_path}")
        
    return task, llm, bestfile

# Function to extract gravity from config.yaml
def get_experiment_config(config_file_path):
    print(f"Reading experiment config from: {config_file_path}")
    with open(config_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        gravity = yaml_data.get('task', {}).get('sim', {}).get('gravity', None)
        
        if gravity:
            print(f"Found gravity: {gravity}")
        else:
            print(f"No gravity found in {config_file_path}")
            
        return gravity

# Function to find the checkpoint file path in BESTFILE.txt
def get_checkpoint_path(root_dir, bestfile, task_name):
    bestfile_txt_path = os.path.join(root_dir, f"{bestfile}.txt")
    print(f"Looking for BESTFILE.txt at: {bestfile_txt_path}")
    
    if os.path.isfile(bestfile_txt_path):
        print(f"BESTFILE.txt found: {bestfile_txt_path}")
        with open(bestfile_txt_path, 'r') as file:
            for line in file:
                # Extract the full second date-time in the format YYYY-MM-DD_HH-MM-SS
                checkpoint_match = re.search(r"runs/(\w+GPT-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})/nn", line)
                if checkpoint_match:
                    # Use the full matched second date-time
                    second_datetime = checkpoint_match.group(1).split('-', 1)[1]
                    # Construct the full path with policy-{second_date_time} and {task_name}GPT-{second_date_time}
                    checkpoint_path = os.path.join(root_dir, f"policy-{second_datetime}", "runs", f"{task_name}GPT-{second_datetime}", "nn", f"{task_name}GPT.pth")
                    print(f"Found checkpoint path: {checkpoint_path}")
                    return checkpoint_path
        print(f"No checkpoint path found in {bestfile_txt_path}")
    else:
        print(f"No BESTFILE.txt found in {bestfile_txt_path}")
    
    return None

# Collect available ROOT directories and relevant information
simulations = []
for subdir in sorted(os.listdir(base_dir)):
    subdir_path = os.path.join(base_dir, subdir)
    print(f"\nChecking directory: {subdir_path}")
    
    if os.path.isdir(subdir_path):
        # Check if eureka.log exists in ROOT
        log_file_path = os.path.join(subdir_path, "eureka.log")
        if os.path.isfile(log_file_path):
            print(f"eureka.log found in {subdir_path}")
            # Parse eureka.log for task, LLM, and BESTFILE
            task, llm, bestfile = parse_log_file(log_file_path)
            if task and llm and bestfile:
                # Retrieve the checkpoint path from BESTFILE.txt
                checkpoint_path = get_checkpoint_path(subdir_path, os.path.splitext(bestfile)[0], task)
                if checkpoint_path:
                    # Look for config.yaml in the parent directory of the nn folder
                    parent_dir = os.path.dirname(os.path.dirname(checkpoint_path))
                    config_file_path = os.path.join(parent_dir, "config.yaml")
                    print(f"Looking for config.yaml at: {config_file_path}")
                    if os.path.isfile(config_file_path):
                        gravity = get_experiment_config(config_file_path)
                    else:
                        gravity = "N/A"
                        print(f"No config.yaml found in {config_file_path}")
                    simulations.append((task, llm, subdir, checkpoint_path, gravity))
                else:
                    print(f"No valid checkpoint path found for task: {task}")
            else:
                print(f"Failed to parse task, LLM, or BESTFILE from {log_file_path}")
        else:
            print(f"No eureka.log found in {subdir_path}")
    else:
        print(f"{subdir_path} is not a directory")

# Display available simulations
if not simulations:
    print("\nNo valid simulations found.")
else:
    print("\nAvailable simulations (sorted by date, oldest first):")
    for idx, (task, llm, date, checkpoint, gravity) in enumerate(simulations, 1):
        print(f"{idx}. Task: {task}, Experiment: {llm}, Date: {date}, Gravity: {gravity}")

    # Prompt user to select a simulation
    selection = int(input("Enter the number of the simulation to test: ")) - 1
    selected_task, selected_llm, selected_date, selected_checkpoint, selected_gravity = simulations[selection]

    # Construct the command
    gravity_str = "[" + ", ".join(map(str, selected_gravity)) + "]" if isinstance(selected_gravity, list) else selected_gravity
    command = [
        "python", "/home/vandriel/Documents/GitHub/Eureka/isaacgymenvs/isaacgymenvs/train.py",
        f"task={selected_task}GPT",
        f"checkpoint={selected_checkpoint}",
        f"+task.sim.gravity={gravity_str}",  # Pass formatted gravity here
        "test=True", "num_envs=4096", "headless=False", "force_render=True"
    ]

    # Execute the command
    print(f"Running test for {selected_task} using {selected_llm} from {selected_date} with Gravity: {selected_gravity}...")
    subprocess.run(command)






# import os
# import re
# import subprocess

# # Base directory where all simulation ROOT directories are stored
# base_dir = "/home/vandriel/Documents/GitHub/Eureka/eureka/outputs/eureka"

# # Function to extract task, LLM, and BESTFILE from eureka.log
# def parse_log_file(log_file_path):
#     task = None
#     llm = None
#     bestfile = None

#     with open(log_file_path, 'r') as file:
#         lines = file.readlines()

#         # Extract task and LLM from the first few lines
#         for line in lines[:10]:  # Adjust if necessary based on actual log length
#             if "Task:" in line:
#                 task_match = re.search(r'Task: (\w+)', line)
#                 if task_match:
#                     task = task_match.group(1)
#             if "Using LLM:" in line:
#                 llm_match = re.search(r'Using LLM: (\w+)', line)
#                 if llm_match:
#                     llm = llm_match.group(1)

#         # Extract BESTFILE from the last few lines
#         for line in lines[-10:]:  # Adjust if necessary based on actual log length
#             bestfile_match = re.search(r'Best Reward Code Path: (\S+)', line)
#             if bestfile_match:
#                 bestfile = bestfile_match.group(1)
#                 break

#     return task, llm, bestfile

# # Function to find the checkpoint file path in BESTFILE.txt
# def get_checkpoint_path(root_dir, bestfile, task_name):
#     bestfile_txt_path = os.path.join(root_dir, f"{bestfile}.txt")
#     if os.path.isfile(bestfile_txt_path):
#         with open(bestfile_txt_path, 'r') as file:
#             for line in file:
#                 # Extract the full second date-time in the format YYYY-MM-DD_HH-MM-SS
#                 checkpoint_match = re.search(r"runs/(\w+GPT-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})/nn", line)
#                 if checkpoint_match:
#                     # Use the full matched second date-time
#                     second_datetime = checkpoint_match.group(1).split('-', 1)[1]
#                     # Construct the full path with policy-{second_date_time} and {task_name}GPT-{second_date_time}
#                     return os.path.join(root_dir, f"policy-{second_datetime}", "runs", f"{task_name}GPT-{second_datetime}", "nn", f"{task_name}GPT.pth")
#     return None

# # Collect available ROOT directories and relevant information
# simulations = []
# for subdir in sorted(os.listdir(base_dir)):
#     subdir_path = os.path.join(base_dir, subdir)
#     if os.path.isdir(subdir_path):
#         # Check if eureka.log exists in ROOT
#         log_file_path = os.path.join(subdir_path, "eureka.log")
#         if os.path.isfile(log_file_path):
#             # Parse eureka.log for task, LLM, and BESTFILE
#             task, llm, bestfile = parse_log_file(log_file_path)
#             if task and llm and bestfile:
#                 # Retrieve the checkpoint path from BESTFILE.txt
#                 checkpoint_path = get_checkpoint_path(subdir_path, os.path.splitext(bestfile)[0], task)
#                 if checkpoint_path:
#                     simulations.append((task, llm, subdir, checkpoint_path))

# # Display available simulations
# if not simulations:
#     print("No valid simulations found.")
# else:
#     print("Available simulations (sorted by date, oldest first):")
#     for idx, (task, llm, date, checkpoint) in enumerate(simulations, 1):
#         print(f"{idx}. Task: {task}, LLM: {llm}, Date: {date}")

#     # Prompt user to select a simulation
#     selection = int(input("Enter the number of the simulation to test: ")) - 1
#     selected_task, selected_llm, selected_date, selected_checkpoint = simulations[selection]

#     # Construct the command
#     command = [
#         "python", "/home/vandriel/Documents/GitHub/Eureka/isaacgymenvs/isaacgymenvs/train.py",
#         f"task={selected_task}GPT",
#         f"checkpoint={selected_checkpoint}",
#         "test=True", "num_envs=4096", "headless=False", "force_render=True"
#     ]

#     # Execute the command
#     print(f"Running test for {selected_task} using {selected_llm} from {selected_date}...")
#     subprocess.run(command)
