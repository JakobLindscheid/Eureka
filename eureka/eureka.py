import hydra
import numpy as np
import datetime
import json
import logging 
import matplotlib.pyplot as plt
import os
from language_model import LanguageModel
# import torch
# torch.cuda.set_per_process_memory_fraction(1.0, 0)
import re
import subprocess
from pathlib import Path
import shutil
import time 
from omegaconf import OmegaConf
from utils.misc import * 
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *

EUREKA_ROOT_DIR = os.getcwd()
# print(f"EUREKA_ROOT_DIR = {EUREKA_ROOT_DIR}")

ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):

    if cfg.eval_human or cfg.evaluate_only:
        logging.warning("Human reward function selected. Skipping Eureka generation.")
        evaluate(cfg, cfg.reward_code_path)
        return
    
    if cfg.use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_username, 
            name=f"{cfg.wandb_name}_main",
            group=cfg.wandb_name,
        )
        wandb.config.update(cfg)

    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    # PVD: Print or log the cfg settings before starting the loop
    # logging.info(f"Configuration settings:\n{OmegaConf.to_yaml(cfg)}")
    # exit()
    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    provider = cfg.provider
    logging.info(f"Using LLM: {model} ({provider})")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    local_model = LanguageModel(model, provider=provider)

    env_name = cfg.env.env_name.lower()
    env_parent = cfg.env_parent # 'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity'
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    # print(f"taskfile = {task_file}")
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    with open("messages.txt", 'w') as f:
        f.write(f"System:\n{initial_system}\n\nUser:\n{initial_user}")

    task_code_string = task_code_string.replace(f"class {task}", f"class {task+suffix}")
    # Create Task YAML files
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 

    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample # if "gpt-3.5" in model else 4
        # chunk_size = cfg.sample if "gpt-3.5" in model else 1 # DEEPSEEK can only handle 1 sample at a time

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = local_model.generate_text(messages, cfg.temperature, chunk_size)
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur["choices"])
            if "usage" in response_cur.keys():
                prompt_tokens = response_cur["usage"]["prompt_tokens"]
                total_completion_token += response_cur["usage"]["completion_tokens"]
                total_token += response_cur["usage"]["total_tokens"]

        """ if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n") """

        # Logging Token Information
        if "usage" in response_cur.keys():
            logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [] 
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```'
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])

            # Add the Eureka Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                continue

            code_runs.append(code_string)
            if env_parent == 'isaac':
                reward_signature = [
                    f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                    f"self.extras['gpt_reward'] = self.rew_buf.mean().item()",
                    f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                ]
            else:
                reward_signature = [f"return {gpt_reward_signature}"]
            indent = " " * 4
            reward_signature = "\n".join([indent*2 + line for line in reward_signature])
            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
            elif "def compute_reward_wrapper(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward_wrapper(self):", "def compute_reward_wrapper(self):\n" + reward_signature)
            else:
                raise NotImplementedError

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')
                if env_parent == 'isaac':
                    file.writelines("from typing import Tuple, Dict" + '\n')
                    file.writelines("import math" + '\n')
                    file.writelines("import torch" + '\n')
                    file.writelines("from torch import Tensor" + '\n')
                    if "@torch.jit.script" not in code_string:
                        code_string = "@torch.jit.script\n" + code_string
                elif gpt_reward_signature.startswith("self") and code_string.startswith("def"):
                    code_string = indent + code_string.replace("\n", f"\n{indent}")
                file.writelines(code_string + '\n')

            os.makedirs(f"iter{iter}/response{response_id}")
            with open(f"iter{iter}/response{response_id}/env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"iter{iter}/response{response_id}/env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()

            # Execute the python file with flags
            rl_filepath = f"iter{iter}/response{response_id}/env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                            'hydra/output=subprocess',  f'hydra.run.dir=./iter{iter}/response{response_id}',
                                            f'task={task}{suffix}', 
                                            # f'wandb_activate={cfg.use_wandb}',
                                            # f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                            f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                            f'max_iterations={cfg.max_iterations}'
                                            ],
                                            stdout=f, stderr=f)
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            # process.wait()
            rl_runs.append(process)

        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []

        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            logging.info(f"Iteration {iter}: Code Run {response_id} completed with return code {rl_run.returncode}")
            rl_filepath = f"iter{iter}/response{response_id}/env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"iter{iter}/response{response_id}/env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read() 
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '' and 'Tensorboard Directory:' in stdout_str:
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break 
                tensorboard_logdir = line.split(':')[-1].strip() 
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                epoch_freq = max(int(max_iterations // 10), 1)

                content += policy_feedback.format(epoch_freq=epoch_freq)

                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["gt_reward"])
                    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in tensorboard_logs:
                    if "/" not in metric:
                        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                        metric_cur_max = max(tensorboard_logs[metric])
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                        
                        if "consecutive_successes" == metric:
                            
                            if cfg.eureka_selection == "max": # --> original method
                                successes.append(metric_cur_max)
                            
                            elif cfg.eureka_selection == "mean":
                                successes.append(metric_cur_mean)
                            
                            elif cfg.eureka_selection == "tail":
                                # compute the length of the tail
                                tail = int(cfg.tail_eval_fraction*len(tensorboard_logs[metric]))
                                
                                if cfg.tail_eval_method == "max":
                                    successes.append(max(tensorboard_logs[metric][-tail:]))
                                elif cfg.tail_eval_method == "mean":
                                    successes.append(sum(tensorboard_logs[metric][-tail:]) / tail)
                                else:
                                    raise NotImplementedError
                            
                            else:
                                raise NotImplementedError
                        
                        metric_cur_min = min(tensorboard_logs[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric 
                            else:
                                metric_name = "task_score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "consecutive_successes" not in tensorboard_logs:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                code_feedbacks.append(code_feedback)
                content += code_feedback  
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 

        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue

        logging.info(f"Iteration {iter}: Code Generation Successes: {successes}")
        
        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes)) if cfg.sample > 1 else 0
        best_content = contents[best_sample_idx]

        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) != DUMMY_FAILURE) / cfg.sample

        # Update the best Eureka Output
        new_best = max_success > max_success_overall
        if new_best:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        if cfg.use_wandb:
            wandb.log({"Max Success": max_success, "Execute Rate": execute_rate, "Max Success Reward Correlation": max_success_reward_correlation})
        logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        with open("messages.txt", 'a') as f:
            f.write(f'\n\nAssistant:\n{responses[best_sample_idx]["message"]["content"]}\n\nUser:\n{best_content}')
        # logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
        # logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
            
        # Plot the success rate
        fig, axs = plt.subplots(2, figsize=(6, 6))
        fig.suptitle(f'{cfg.env.task}')

        x_axis = np.arange(len(max_successes))

        axs[0].plot(x_axis, np.array(max_successes))
        axs[0].set_title("Max Success")
        axs[0].set_xlabel("Iteration")

        axs[1].plot(x_axis, np.array(execute_rates))
        axs[1].set_title("Execute Rate")
        axs[1].set_xlabel("Iteration")

        fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
        # np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

        if new_best or cfg.ea_selection == ",": # 
            # TODO: This seems strange, we should discuss it.
            if len(messages) == 2:
                messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
                messages += [{"role": "user", "content": best_content}]
            else:
                assert len(messages) == 4
                messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
                messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)
    
    if cfg.use_wandb:
        wandb.save("messages.json")
        wandb.save("messages.txt")
        wandb.save(max_reward_code_path.replace(".py", "_rewardonly.py"))

    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    
    evaluate(cfg, max_reward_code_path)

def evaluate(cfg, max_reward_code_path):
    
    if cfg.use_wandb:
        import wandb
        if wandb.run is None:
            wandb.init(
                project=cfg.wandb_project, 
                entity=cfg.wandb_username, 
                name=f"{cfg.wandb_name}_evaluation",
                group=cfg.wandb_name,
            )
            wandb.config.update(cfg)
            
            if not cfg.eval_human:
                wandb.save(max_reward_code_path)
    
    task = cfg.env.task
    suffix = cfg.suffix if not cfg.eval_human else ""
    env_name = cfg.env.env_name.lower()
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"
    
    # Evaluate the best reward code many times    
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    
    if not cfg.eval_human:
        try:
            shutil.copy(max_reward_code_path, output_file)
        except shutil.SameFileError:
            pass
    
    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()

        # Execute the python file with flags
        os.mkdir(f"eval{i}")
        rl_filepath = f"eval{i}/reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                        'hydra/output=subprocess', f'hydra.run.dir=./eval{i}',
                                        f'task={task}{suffix}', 
                                        f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}', 
                                        f'wandb_name={cfg.wandb_name}_eval{i}', f'wandb_group={cfg.wandb_name}',
                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}'
                                        ],
                                        stdout=f, stderr=f)

        block_until_training(rl_filepath)
        # process.wait()
        eval_runs.append(process)
    
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()

        if cfg.eval_episodes > 0:
            rl_filepath = f"eval{i}/reward_code_eval{i}.txt"

            with open(rl_filepath, 'r') as f:
                stdout_str = f.read() 
            lines = stdout_str.split('\n')
            for line in lines:
                if line.startswith('Network Directory:'):
                    break 

            network_dir = line.split(':')[-1].strip()
            checkpoint = f"{network_dir}/{os.listdir(network_dir)[0]}"
            logging.info(f"Evaluating checkpoint: {checkpoint}")

            eval_episodes = []
            for j in range(cfg.eval_episodes):
                set_freest_gpu()

                # Execute the python file with flags
                os.mkdir(f"eval{i}/episode{j}")
                path = f"eval{i}/episode{j}/eval_episode{j}.txt"
                with open(path, 'w') as f:
                    process = subprocess.Popen(['python', '-u', f"{ISAAC_ROOT_DIR}/train.py", 
                                                'hydra/output=subprocess', f'hydra.run.dir=./eval{i}/episode{j}',
                                                f'checkpoint={checkpoint}', f'test=True',
                                                f'task={task}{suffix}', f'wandb_activate=False',
                                                f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={j}',
                                                ],
                                                stdout=f, stderr=f)
                    
                # block until start
                while True:
                    with open(path, 'r') as file:
                        rl_log = file.read()
                        if "reward:" in rl_log or "Traceback" in rl_log:                            
                            break
                        time.sleep(1)
                
                eval_episodes.append(process)
    
    reward_code_final_successes = {"max": [], "mean": [], "tail": []}
    reward_code_correlations_final = []

    if cfg.use_wandb:
        wandb_table = wandb.Table(columns=["run", "Max Success", "Mean Success", "Tail Success", "Correlation"])
    for i, rl_run in enumerate(eval_runs):
        # rl_run.communicate()
        if cfg.eval_episodes == 0:
            rl_filepaths = [f"eval{i}/reward_code_eval{i}.txt"]
        else:
            rl_filepaths = []
            for j, eval_ep in enumerate(eval_episodes):
                eval_ep.communicate()
                rl_filepaths.append(f"eval{i}/episode{j}/eval_episode{j}.txt")
        
        for rl_filepath in rl_filepaths:
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read() 
            except Exception as e:
                logging.error(f"Error opening {rl_filepath}: {e}")
                continue
            
            traceback_msg = filter_traceback(stdout_str)
            
            if traceback_msg == '' and 'Tensorboard Directory:' in stdout_str:
                lines = stdout_str.split('\n')
                for line in lines:
                    if line.startswith('Tensorboard Directory:'):
                        break 
                tensorboard_logdir = line.split(':')[-1].strip() 
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                
                max_success = max(tensorboard_logs['consecutive_successes'])
                mean_success = sum(tensorboard_logs['consecutive_successes']) / len(tensorboard_logs['consecutive_successes'])
                tail = int(cfg.tail_eval_fraction*len(tensorboard_logs['consecutive_successes']))
                tail_mean = sum(tensorboard_logs['consecutive_successes'][-tail:]) / tail
                
                # reward_code_final_successes.append(max_success)
                reward_code_final_successes["max"].append(max_success)
                reward_code_final_successes["mean"].append(mean_success)
                reward_code_final_successes["tail"].append(tail_mean)

                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                    gt_reward = np.array(tensorboard_logs["gt_reward"])
                    gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_code_correlations_final.append(reward_correlation)

                else:
                    reward_correlation = None

                if cfg.use_wandb:
                    run_name = f"eval{i}_episode{rl_filepath.split('episode')[1][0]}" if "episode" in rl_filepath else f"eval{i}"
                    wandb_table.add_data(run_name, max_success, mean_success, tail_mean, reward_correlation)

            else:
                splitted = rl_filepath.split("/")
                name = splitted[0] if len(splitted) == 1 else f"{splitted[0]}_{splitted[1]}"
                logging.error(f"{name} failed.")

    
    if cfg.use_wandb:
        wandb.log({
            "Final Max-Success Mean": np.mean(reward_code_final_successes["max"]), 
            "Final Mean-Success Mean": np.mean(reward_code_final_successes["mean"]),
            "Final Tail-Success Mean": np.mean(reward_code_final_successes["tail"]),
            "Final Correlation Mean": np.mean(reward_code_correlations_final)
        })
        wandb.log({"eval_metrics_raw": wandb_table})

    print()
    logging.info(f"Final Max-Success Mean: {np.mean(reward_code_final_successes['max'])}, Std: {np.std(reward_code_final_successes['max'])}")
    logging.info(f"Final Max-Success Raw: {reward_code_final_successes['max']}")
    print()
    logging.info(f"Final Mean-Success Mean: {np.mean(reward_code_final_successes['mean'])}, Std: {np.std(reward_code_final_successes['mean'])}")
    logging.info(f"Final Mean-Success Raw: {reward_code_final_successes['mean']}")
    print()
    logging.info(f"Final Tail-Success Mean: {np.mean(reward_code_final_successes['tail'])}, Std: {np.std(reward_code_final_successes['tail'])}")
    logging.info(f"Final Tail-Success Raw: {reward_code_final_successes['tail']}")
    print()
    logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}")
    logging.info(f"Final Correlation Raw: {reward_code_correlations_final}")
    # np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()