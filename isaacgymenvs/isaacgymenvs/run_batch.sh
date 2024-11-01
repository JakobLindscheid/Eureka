#!/bin/bash

# List of gravity values
gravity_values=(
    "[0.0, 0.0, -9.81]"
    "[0.855, 0.0, -9.77]"
    "[2.54, 0.0, -9.475]"
    "[-0.855, 0.0, -9.77]"
    "[-2.54, 0.0, -9.475]"
)

# List of experiment suffixes for each gravity value
experiment_suffixes=(
    "flat"
    "down05"
    "down20"
    "up05"
    "up20"
)

# List of tasks
tasks=("Humanoid")

# Number of repetitions
num_repeats=5

# Outer loop for each task
for task in "${tasks[@]}"
do
    echo "Running simulations for task: $task"
    
    # Inner loop for each gravity value and experiment suffix
    for i in "${!gravity_values[@]}"
    do
        gravity="${gravity_values[$i]}"
        experiment_suffix="${experiment_suffixes[$i]}"
        
        # Construct the experiment name as {task}_{suffix}
        experiment="${task}_${experiment_suffix}"
        
        echo "Running simulation with gravity: $gravity and experiment: $experiment for task: $task"
        
        # Run each simulation 10 times
        for ((run=1; run<=num_repeats; run++))
        do
            echo "Running repetition $run for $experiment"
            
            # Call train.py with the overridden gravity value, experiment name, task, and randomize flag
            python train.py task.sim.gravity="$gravity" experiment="$experiment" task="$task" task.task.randomize=True
            
            echo "Completed repetition $run for $experiment"
        done
        
    done
    
done
