FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG PROJECT=model_training

RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx 

RUN apt-get install -y \
    graphviz \
    xdg-utils 

RUN apt-get install -y \
    python3-pip \
    python-is-python3
    
# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY shared/pyproject.toml shared/poetry.lock* /app/shared/
RUN cd /app/shared && poetry install --only main
ENV PYTHONPATH=/app/shared:$PYTHONPATH

# Copy only the pyproject.toml and poetry.lock* to install dependencies
COPY ${PROJECT}/pyproject.toml ${PROJECT}/poetry.lock* /app/

# Install dependencies
RUN cd /app && poetry install --only main

# Install non-poetry dependencies
RUN pip install flash-attn 

WORKDIR /app
