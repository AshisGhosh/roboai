FROM robo_ai-robosuite-simulation

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake 

RUN pytorch pytorch::torchvision --yes


WORKDIR /
RUN git clone https://github.com/ARISE-Initiative/robomimic.git
WORKDIR /robomimic
RUN pip install -e .

# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

# Copy only the pyproject.toml and poetry.lock* to install dependencies
COPY pyproject.toml poetry.lock* /app/

WORKDIR /app

# Install dependencies
RUN poetry install --no-dev


# Use the entrypoint script to activate the Conda environment
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]