# setup
udocker pull mcr.microsoft.com/vscode/devcontainers/python:0-3.8
udocker create --name=eureka mcr.microsoft.com/vscode/devcontainers/python:0-3.8
udocker setup --nvidia --force eureka
# for cmd: udocker run -v ~/Eureka:/workspace/Eureka -w /workspace/Eureka/eureka eureka /bin/bash

# -v option to mount the Eureka directory to the container
# eureka installation steps
udocker run -v ~/Eureka:/workspace/Eureka eureka /bin/bash -c 'cd /workspace && \
    wget https://developer.nvidia.com/isaac-gym-preview-4 -O IsaacGym_Preview_4_Package.tar.gz && \
    tar -xvf IsaacGym_Preview_4_Package.tar.gz && \
    pip install -e isaacgym/python && \
    cd Eureka && pip install -e . && \
    pip install -e isaacgymenvs && \
    pip install -e rl_games && \
    pip install -r requirements.txt && \
    ln -s /usr/local/lib/python3.8/site-packages/torch/lib/libnvrtc-builtins.so /usr/local/lib/python3.8/site-packages/torch/lib/libnvrtc-builtins.so.11.1'
