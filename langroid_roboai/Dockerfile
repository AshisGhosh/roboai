FROM python:3.11-slim

# Install Poetry
# RUN conda run -n robot_env pip install poetry
# RUN conda run -n robot_env poetry config virtualenvs.create false
RUN pip install poetry
RUN poetry config virtualenvs.create false

# Copy only the pyproject.toml and poetry.lock* to install dependencies
COPY pyproject.toml poetry.lock* /app/

WORKDIR /app

# Install dependencies
# RUN conda run -n robot_env poetry install --no-dev
RUN poetry install --no-dev

# Use the entrypoint script to activate the Conda environment
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]