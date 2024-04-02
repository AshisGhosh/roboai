FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6

# RUN conda install python=3.10 -y

# Install grasp_det_seg_cnn
WORKDIR /
RUN git clone https://github.com/stefan-ainetter/grasp_det_seg_cnn.git
WORKDIR /grasp_det_seg_cnn
RUN pip install -r requirements.txt
RUN pip install -e .

WORKDIR /

# RUN apt-get install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y python3.10

# RUN python3.10 --version

# RUN wget https://bootstrap.pypa.io/get-pip.py && \
#     python3.10 get-pip.py && \
#     rm get-pip.py

# RUN conda create -n api_env python=3.10 -y

# SHELL ["/bin/bash", "-c"]
# RUN conda create -n api_env python=3.10 -y && \
#     echo "source activate api_env" > ~/.bashrc && \
#     source ~/.bashrc && \
#     pip install poetry && \
#     poetry config virtualenvs.create false
# RUN conda activate api_env
RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* /app/

WORKDIR /app
RUN poetry install --no-dev


