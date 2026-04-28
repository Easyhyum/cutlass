# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    tmux build-essential wget curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

COPY . /workspace

RUN pip uninstall -y torch torchvision torchaudio

RUN pip install -r /workspace/requirements_torch.txt

# RUN cd /workspace/vllm && \ 
#     export VLLM_VERSION=0.13.0.dev \ 
#     SETUPTOOLS_SCM_PRETEND_VERSION=0.13.0.dev MAX_JOBS=16 pip install -e .
# RUN cd /workspace/vllm && MAX_JOBS=16 pip install -e .

RUN chmod +x .

CMD ["bash"]
