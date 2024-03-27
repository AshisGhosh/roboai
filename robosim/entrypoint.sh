#!/bin/bash

# Activate the Conda environment
echo "Activating Conda environment..."
source activate robot_env

# Execute the command passed to the Docker container
# This allows you to specify the command in the docker-compose.yml
echo "Executing command: $@"
exec "$@"