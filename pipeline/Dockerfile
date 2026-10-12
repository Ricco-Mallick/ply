# Ply pipeline image for RunPod (or any NVIDIA host with Docker + the NVIDIA container toolkit).
#   docker build -t ply-pipeline --build-arg TORCH_CUDA_ARCH_LIST="8.9" pipeline/
#   docker run --gpus all --rm -v $PWD/work:/work ply-pipeline run /work/walk.mp4 --out /work/lab2
#
# CUDA 12.8 devel base: nvcc is needed to compile gsplat / fused-ssim / ppisp against torch cu128.
# COLMAP itself comes prebuilt from the pycolmap-cuda12 4.2.1 wheel, so no C++ build here.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ARG MAX_JOBS=8
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    PLY_GSPLAT_DIR=/opt/gsplat \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/node/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git ffmpeg curl ca-certificates xz-utils build-essential \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# Node 22 LTS for splat-transform (warm the npx cache so export works offline)
RUN curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz | tar -xJ -C /opt \
    && mv /opt/node-v22.14.0-linux-x64 /opt/node \
    && npx -y @playcanvas/splat-transform@3.10.1 --version || true

WORKDIR /opt/ply/pipeline
COPY pyproject.toml ./
COPY plypipe ./plypipe
COPY scripts ./scripts
RUN PYTHON=/opt/venv/bin/python bash scripts/install_gpu.sh

WORKDIR /work
ENTRYPOINT ["python", "-m", "plypipe"]
CMD ["--help"]
