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
## Access a terminal in the container
-v mounts the Eureka directory in the container  
-w specifies the working directory
```
udocker run -v ~/Eureka:/workspace/Eureka -w /workspace/Eureka/eureka eureka /bin/bash
```
## Run a Eureka experiment:
--env to pass an environment variable like API keys  
Modify arguments in the python call.
```
sh run.sh python eureka.py env_parent=isaac env=humanoid use_wandb=True
```

## Evaluate a specified environment (for example with human reward function):
Set suffix to "" when evaluating the human reward, leave unchanged otherwise.
```
sh run.sh python eureka.py env_parent=isaac env=humanoid evaluate_only=True reward_code_path=/path/to/env.py suffix="" use_wandb=True
```

## Capture a video of a specified training checkpoint:
```
sh run.sh python isaacgymenvs/train.py test=True capture_video=True headless=False force_render=True task=ShadowHandSpinGPT checkpoint=path/to/ckpt.pth
```