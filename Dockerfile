# This is a platform Dockerfile fragment: the platform supplies the base image
# and runtime command.
ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda

ENV CONDA_DIR=${CONDA_DIR} \
    PATH=${CONDA_DIR}/envs/videoagent/bin:${CONDA_DIR}/bin:${PATH} \
    CONDA_DEFAULT_ENV=videoagent \
    MODEL_BASE_PATH=/models \
    TOOL_STAR_MODEL_BASE_PATH=/models \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        bzip2 \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda config --system --set channel_priority flexible

WORKDIR /workspace

# The lock-style environment file is the source of truth for the recovered
# Python environment.  Models and datasets are mounted at runtime.
COPY environment.yml ./
RUN conda env create --file environment.yml \
    && conda run --no-capture-output -n videoagent python -c \
        "import cv2, faster_whisper, pydantic_settings, ray, torch, transformers" \
    && conda clean --all --yes

COPY . ./
