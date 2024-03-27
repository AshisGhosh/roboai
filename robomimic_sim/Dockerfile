FROM robo_ai-robosuite-simulation

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake 

RUN conda install -n robot_env pytorch pytorch::torchvision --yes


WORKDIR /
RUN git clone https://github.com/ARISE-Initiative/robomimic.git
WORKDIR /robomimic
RUN conda run -n robot_env pip install -e .

WORKDIR /app

# Use the entrypoint script to activate the Conda environment
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]