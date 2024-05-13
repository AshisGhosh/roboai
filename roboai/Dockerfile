FROM python:3.11-slim

ARG PROJECT=roboai


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
    
# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY shared/pyproject.toml shared/poetry.lock* /app/shared/
RUN cd /app/shared && poetry install --no-dev
ENV PYTHONPATH=/app/shared:$PYTHONPATH

# Copy only the pyproject.toml and poetry.lock* to install dependencies
COPY ${PROJECT}/pyproject.toml ${PROJECT}/poetry.lock* /app/

# Install dependencies
RUN cd /app && poetry install --no-dev

WORKDIR /app
ENV PYTHONPATH=/app/roboai:$PYTHONPATH

# Use the entrypoint script to activate the Conda environment
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]