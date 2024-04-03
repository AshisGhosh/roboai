# Use an official Miniconda3 base image
FROM continuumio/miniconda3

ARG PROJECT=robosim

# Set the Python version
ARG PYTHON_VERSION=3.10

# Set the MuJoCo version
ARG MUJOCO_VERSION=3.1.3

# Install system dependencies for evdev
# For Debian/Ubuntu, installing generic Linux headers
RUN apt-get update && apt-get install -y \
    linux-headers-amd64 \
    gcc \
    libc-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libosmesa6

# Update Conda and install pip in the base environment
RUN conda update -n base -c defaults conda && \
    conda create -n robot_env python=$PYTHON_VERSION pip && \
    conda clean -afy

# Download and install MuJoCo binaries (required by Robosuite)
RUN mkdir -p /root/.mujoco && \
    wget https://github.com/deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -xzf mujoco.tar.gz -C /root/.mujoco && \
    rm mujoco.tar.gz

# Assuming MuJoCo is extracted with the version in its directory name, adjust if different
ENV MUJOCO_PY_MJPRO_PATH=/root/.mujoco/mujoco-${MUJOCO_VERSION}
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco-${MUJOCO_VERSION}/bin:${LD_LIBRARY_PATH}


# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libosmesa6

# Install Robosuite
RUN conda run -n robot_env pip install robosuite

# Copy the entrypoint script into the image and make sure it's executable
RUN mkdir -p /usr/local/bin
COPY ${PROJECT}/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Install Poetry
RUN conda run -n robot_env pip install poetry
RUN conda run -n robot_env poetry config virtualenvs.create false

# Copy only the pyproject.toml and poetry.lock* to install dependencies
COPY ${PROJECT}/pyproject.toml ${PROJECT}/poetry.lock* /app/

WORKDIR /app

# Install dependencies
RUN conda run -n robot_env poetry install --no-dev

COPY shared/pyproject.toml shared/poetry.lock* /app/shared/
RUN cd /app/shared && conda run -n robot_env poetry install --no-dev


# Use the entrypoint script to activate the Conda environment
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# The CMD can be overridden by specifying a command in docker-compose.yml
CMD ["echo", "Specify a command in docker-compose.yml"]
