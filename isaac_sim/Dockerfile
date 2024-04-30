FROM nvcr.io/nvidia/isaac-sim:2023.1.1

ENV DEBIAN_FRONTEND=noninteractive

RUN ./python.sh -m pip install \
        fastapi \
        uvicorn \
        python-multipart


# Patch omniverse
COPY ./isaac_sim/omniverse_patch/nucleus.py /isaac-sim/exts/omni.isaac.core/omni/isaac/core/utils/nucleus.py

RUN apt-get update && apt-get install -y \
                 libglib2.0-0 \
                 libsm6 \
                 libxrender1 \
                 libxext6

RUN ./python.sh -m pip install \
        opencv-python

# Dependencies for /shared
RUN ./python.sh -m pip install \
        httpx \
        pillow \
        litellm

# Install ROS2
RUN apt update && apt install -y locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN apt install -y software-properties-common
RUN add-apt-repository universe
RUN apt install -y curl
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update
RUN apt install -y ros-humble-desktop \
                ros-humble-camera-info-manager \
                ros-humble-rosbag2 \
                ros-humble-rosbag2-compression-zstd \
                ros-humble-rosbag2-cpp \
                ros-humble-rosbag2-py \
                ros-humble-rosbridge-suite \
                ros-humble-rqt-graph \
                ros-humble-rqt-reconfigure \
                ros-humble-rviz-common \
                ros-humble-rviz-default-plugins \
                ros-humble-vision-msgs

RUN apt install -y python3-colcon-common-extensions \
                python3-rosdep \
                python3-vcstool \
                python3-rosinstall-generator

WORKDIR /root
RUN apt-get install -y git
RUN git clone https://github.com/NVIDIA-Omniverse/IsaacSim-ros_workspaces.git
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
         cd IsaacSim-ros_workspaces/humble_ws && \
         colcon build
RUN echo "source /root/IsaacSim-ros_workspaces/humble_ws/install/setup.bash" >> /root/.bashrc
WORKDIR /isaac-sim

