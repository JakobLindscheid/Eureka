# Setup for DS Lab servers
## Install udocker
```
pip install udocker
udocker install
```

## Run setup script

Assuming the Eureka repository is cloned in your home directory.

```
cd ~/Eureka
sh ./setup.sh
```

# How to use

Run any command on the container by calling run.sh:
```
sh run.sh echo test
```

## Access a terminal in the container
Just don't pass a command.
```
sh run.sh
```
## Run a Eureka experiment:
Add API keys as environment variables. The run file automatically transfers the `OPENAI_API_KEY` and the `GROQ_KEY` variables to the container. You can easily add more in `run.sh`.  
Modify arguments in the python call.
```
sh run.sh python eureka.py env_parent=isaac env=humanoid use_wandb=True
```

## Evaluate a specified environment (for example with human reward function):
Set suffix to "" when evaluating the human reward, leave unchanged otherwise.
```
sh run.sh python eureka.py env_parent=isaac env=humanoid evaluate_only=True reward_code_path=/path/to/env.py suffix="" use_wandb=True
```