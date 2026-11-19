# UniVLA environment: Ubuntu 22.04 + CUDA 12.1 + Python 3.10 + torch 2.2.0
#
# The image holds ONLY the environment (system packages, Python libraries,
# LIBERO). Code and data are bind-mounted from the host when the container
# is started (see launch_container.sh):
#   ~/projects/univla -> /workspace/univla
#   ~/data/univla     -> /data
#
# Build from the repository root:
#   docker build -t univla:cu121 -f docker/Dockerfile .

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1

# --- System packages (Python, tools, OpenGL/EGL libraries for headless MuJoCo rendering) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        git wget curl vim tmux ca-certificates build-essential \
        libegl1 libgl1 libgles2 libglib2.0-0 libosmesa6 libglfw3 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Lets libglvnd find NVIDIA's EGL driver (MuJoCo/LIBERO renders through EGL on the GPU)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN python -m pip install --upgrade pip

# --- PyTorch 2.2.0 + CUDA 12.1 (the version used by the authors) ---
RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# --- Repository dependencies (the constraints file stops pip from replacing torch) ---
COPY docker/requirements.txt /tmp/requirements.txt
RUN printf "torch==2.2.0\ntorchvision==0.17.0\n" > /tmp/constraints.txt && \
    pip install -r /tmp/requirements.txt -c /tmp/constraints.txt

# --- Flash Attention 2.5.5 from a prebuilt wheel (no compilation needed) ---
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# --- LIBERO simulation benchmark ---
# Answering "n" to the first-import prompt writes the default config (~/.libero/config.yaml)
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO && \
    pip install --no-deps -e /opt/LIBERO
RUN echo n | python -c "import libero.libero" && test -f /root/.libero/config.yaml

# --- Runtime environment ---
# PYTHONPATH: the repo (mounted at runtime) is importable without `pip install -e .`
ENV PYTHONPATH=/workspace/univla \
    HF_HOME=/data/cache/hf \
    TORCH_HOME=/data/cache/torch \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=all \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /workspace/univla
CMD ["bash"]
