FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    zip \
    # ffmpeg \
    ffmpeg \
    x264 \
    # python build dependencies \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH}
WORKDIR ${HOME}/app

RUN curl https://pyenv.run | bash
ENV PATH=${HOME}/.pyenv/shims:${HOME}/.pyenv/bin:${PATH}
ARG PYTHON_VERSION=3.9.16
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    pyenv rehash && \
    pip install --no-cache-dir -U pip setuptools wheel

RUN pip install --no-cache-dir -U torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
COPY --chown=1000 requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir -U kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html
RUN pip install --no-cache-dir -U gradio==3.30.0

COPY --chown=1000 . ${HOME}/app
RUN cd TEXTurePaper && patch -p1 < ../patch
ENV PYTHONPATH=${HOME}/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces
CMD ["python", "app.py"]
