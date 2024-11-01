sim='Humanoid' 
model='DS' #GPT or DS
#model_path='2024-09-23_21-46-02/policy-2024-09-23_22-05-34/runs/QuadcopterGPT-2024-09-23_22-05-34/nn'
#model_path='2024-10-01_14-35-40/policy-2024-10-01_19-46-08/runs/AntGPT-2024-10-01_19-46-08/nn' # original ant, success = 10.28
#model_path='2024-10-02_15-49-34/policy-2024-10-02_16-39-19/runs/AntGPT-2024-10-02_16-39-19/nn' # limp ant, leg and slope, fails
#model_path='2024-10-02_17-06-23/policy-2024-10-02_20-38-16/runs/AntGPT-2024-10-02_20-38-16/nn' # limp ant, leg, fails
#model_path='2024-10-04_14-11-53/policy-2024-10-04_14-33-20/runs/AntGPT-2024-10-04_14-33-20/nn' # limp ant, slope, success = 1.53
#model_path='2024-10-04_18-15-58/policy-2024-10-04_18-17-37/runs/CartpoleDS-2024-10-04_18-17-38/nn' # cartpole, success, deepseek 
#model_path='2024-10-04_23-07-50/policy-2024-10-04_23-10-21/runs/BallBalanceDS-2024-10-04_23-10-22/nn' # ballbalance, success, deepseek
#model_path='2024-10-06_10-52-49/policy-2024-10-06_11-11-39/runs/HumanoidDS-2024-10-06_11-11-39/nn' # humanoid, success = 7.05
model_path='2024-10-06_15-05-55/policy-2024-10-06_21-01-42/runs/HumanoidGPT-2024-10-06_21-01-43/nn' # humanoid, success = 7.31
eureka_path="/home/vandriel/Documents/GitHub/Eureka/"
show_path="${eureka_path}eureka/outputs/eureka/${model_path}/"
python $eureka_path/isaacgymenvs/isaacgymenvs/train.py test=True headless=False force_render=True task=$sim checkpoint="${show_path}${sim}${model}.pth"
