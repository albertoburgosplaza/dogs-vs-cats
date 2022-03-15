FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y \
    build-essential \
    cmake \
    curl \
    gcc \
    git \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    locales \
    net-tools \
    wget \
    libgl1-mesa-glx \
    zsh \
    parallel \
    sudo \
    python3-venv \
    python3-dev

RUN rm -rf /var/lib/apt/lists/*

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV LANG=en_US.UTF-8
RUN locale-gen en_US.UTF-8

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID ml
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID ml

RUN usermod -a -G sudo ml 
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ml
WORKDIR /home/ml

RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
ENV ZSH_THEME agnoster
ENV SHELL /bin/zsh