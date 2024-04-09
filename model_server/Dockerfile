# Use an official lightweight Python image.
FROM python:3.12-slim

ARG PROJECT=model_server

# Set the working directory in the container
WORKDIR /app

# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

# Assuming shared dependencies are used by model_server as well
COPY shared/pyproject.toml shared/poetry.lock* /app/shared/
RUN cd /app/shared && poetry install --no-dev

# Copy only the pyproject.toml and poetry.lock* for model_server to install dependencies
COPY ${PROJECT}/pyproject.toml ${PROJECT}/poetry.lock* /app/

# Install model_server dependencies
RUN poetry install --no-dev

# The source code will be mounted via Docker Compose, so no need to COPY
# COPY model_server /app
# COPY shared /app/shared

# The command to run the app will be specified in docker-compose.yml
