FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6

# Install grasp_det_seg_cnn
WORKDIR /
RUN git clone https://github.com/stefan-ainetter/grasp_det_seg_cnn.git
WORKDIR /grasp_det_seg_cnn
RUN pip install -r requirements.txt
RUN pip install -e .

WORKDIR /

RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* /app/

WORKDIR /app
RUN poetry install --no-dev


