# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import logging
import os
import datetime
import numpy as np

try:
    import isaacgym
except ImportError:
    logging.warning("Isaac Gym is not installed. If you are running an Isaac Gym task, please install it.")

import hydra
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym
import sys 
import shutil
from pathlib import Path
import subprocess
import json
import time

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']
    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict

# these functions are copied from eureka/utils/misc.py
def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    gpu_found = False
    while not gpu_found:
        sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str, _ = sp.communicate()
        gpustats = json.loads(out_str.decode('utf-8'))
        # Find GPU with most free memory
        # freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

        for freest_gpu in sorted(gpustats['gpus'], key=lambda x: x['memory.used']):

            free_memory = freest_gpu['memory.total'] - freest_gpu['memory.used']
            if len(freest_gpu['processes']) == 0:
                gpu_found = True
                break
            else:
                largest_process = max(freest_gpu['processes'], key=lambda x: x['gpu_memory_usage'])
                if (
                    largest_process['gpu_memory_usage'] < free_memory and # if the new process will not exceed the free memory
                    freest_gpu['utilization.gpu'] + freest_gpu['utilization.gpu']/len(freest_gpu['processes']) <= 95 # if adding a new process will not exceed 95% utilization
                ):
                    gpu_found = True
                    break
        
        if not gpu_found:
            time.sleep(5)

    return freest_gpu['index']


@hydra.main(config_name="config", config_path="./cfg", version_base="1.1")
def launch_rlg_hydra(cfg: DictConfig):

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs
    """ # Initialize the log file by creating or clearing it at the start of training
    log_file_path = f"{ROOT_DIR}/consecutive_successes_log.txt"
    with open(log_file_path, "w") as f:
        f.write("")  # This clears the file if it already exists or creates it if it doesn't """


    # time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = cfg.wandb_name # f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    rank = int(os.getenv("LOCAL_RANK", "0"))
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg.train.params.config.multi_gpu = cfg.multi_gpu


    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            if cfg.test:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0),
                    video_length=cfg.capture_video_len,
                )
            else:
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: (step % cfg.capture_video_freq == 0) and (step > 0),
                    video_length=cfg.capture_video_len,
                )
        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })
    
    # Save the environment code!
    try:
        output_file = f"{ROOT_DIR}/tasks/{cfg.task.env.env_name.lower()}.py"
        shutil.copy(output_file, f"env.py")
    except:
        import re
        def camel_to_snake(name):
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        output_file = f"{ROOT_DIR}/tasks/{camel_to_snake(cfg.task.name)}.py"

        shutil.copy(output_file, f"env.py")

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    observers = [RLGPUAlgoObserver()]

    if cfg.wandb_activate and rank ==0 :

        import wandb
        
        # initialize wandb only once per horovod run (or always for non-horovod runs)
        wandb_observer = WandbAlgoObserver(cfg)
        observers.append(wandb_observer)

    rlg_config_dict['params']['config']['full_experiment_name'] = cfg.wandb_name

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    statistics = runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint' : cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })

    # dump config dict
    experiment_dir = os.path.join('runs', os.listdir('runs')[0])
    print("Tensorboard Directory:", Path.cwd() / experiment_dir / "summaries")
    if not cfg.test:
        print("Network Directory:", Path.cwd() / experiment_dir / "nn")

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

        checkpoint = os.listdir(f"{experiment_dir}/nn")[0]
        if cfg.wandb_activate and rank == 0:
            wandb.save(str(Path.cwd() / experiment_dir / "nn" / checkpoint))
        if cfg.eval_episodes > 0:
            print(f"Checkpoint: {checkpoint}")

            eval_runs = []            
            for i in range(cfg.eval_episodes):
                set_freest_gpu()

                # Execute the python file with flags
                os.mkdir(f"episode{i}")
                path = f"episode{i}/eval_episode{i}.txt"
                with open(path, 'w') as f:
                    process = subprocess.Popen(['python', '-u', f"{ROOT_DIR}/train.py", 
                                                'hydra/output=subprocess', f'hydra.run.dir=./episode{i}',
                                                f'checkpoint={Path.cwd() / experiment_dir / "nn" / checkpoint}', f'test=True',
                                                f'task={cfg.task_name}',
                                                f'wandb_activate=False',
                                                # f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}', 
                                                # f'wandb_name={cfg.wandb_name}_eval{i}', f'wandb_group={cfg.wandb_name}',
                                                f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}',
                                                ],
                                                stdout=f, stderr=f)
                # block until start
                while True:
                    with open(path, 'r') as file:
                        rl_log = file.read()
                        if "reward:" in rl_log or "Traceback" in rl_log:
                            break
                
                eval_runs.append(process)

        for process in eval_runs:
            process.communicate()
    
    # After runner.run() completes, read the log file for consecutive successes
    try:
        with open(f"{ROOT_DIR}/consecutive_successes_log.txt", "r") as f:
            successes = [float(line.strip()) for line in f]
            slice_index = int(len(successes) * 0.9)

        # Redirect output to a file
        with open(f"{ROOT_DIR}/output_batch.txt", "a") as output_file:
            if successes:
                success_message = f"Test success: {np.mean(successes[slice_index:]):.2f} pm {np.std(successes[slice_index:]):.2f}, Max: {max(successes[slice_index:]):.2f}\n"
                print(success_message)  # This will still print to the console
                output_file.write(success_message)  # Write to file
            else:
                no_success_message = "No successes logged.\n"
                print(no_success_message)  # This will still print to the console
                output_file.write(no_success_message)  # Write to file

    except FileNotFoundError:
        print("Log file not found. No successes were logged.")

    if cfg.wandb_activate and rank == 0:
        wandb.finish()
        
if __name__ == "__main__":
    launch_rlg_hydra()
