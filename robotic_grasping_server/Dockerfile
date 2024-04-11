FROM python:3.11-slim

ARG PROJECT=robotic_grasping_server

RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx

RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* /app/
RUN cd /app && poetry install --no-dev

RUN git clone https://github.com/skumra/robotic-grasping.git

RUN cd robotic-grasping && pip install -r requirements.txt 

ENV PYTHONPATH "${PYTHONPATH}:/robotic-grasping"

WORKDIR /app