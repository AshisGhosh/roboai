FROM nvcr.io/nvidia/isaac-sim:2023.1.1

ENV DEBIAN_FRONTEND=noninteractive

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

COPY ./isaac_sim/humble_ws /isaac-sim/humble_ws
RUN chmod -R 777 /isaac-sim/humble_ws
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
         cd humble_ws && \
         colcon build
RUN echo "source /isaac-sim/humble_ws/install/setup.bash" >> /root/.bashrc

RUN apt-get update && apt-get install -y \
                ros-humble-depth-image-proc 


RUN apt-get install git -y

ARG ROS_DISTRO=humble                
RUN mkdir -p ~/ws_moveit2/src
WORKDIR /root/ws_moveit2/src
# Clone the moveit2 repository
RUN git clone https://github.com/moveit/moveit2.git -b main
RUN for repo in moveit2/moveit2.repos $(f="moveit2/moveit2_$ROS_DISTRO.repos"; test -r $f && echo $f); do \
        vcs import < "$repo"; \
        done
RUN rosdep init && rosdep update
RUN rosdep install -r --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y
RUN source /opt/ros/humble/setup.bash && \
        cd ~/ws_moveit2 && \
        colcon build --event-handlers desktop_notification- status- --cmake-args -DCMAKE_BUILD_TYPE=Release
RUN echo "source ~/ws_moveit2/install/setup.bash" >> /root/.bashrc
# TODO: debug why colcon build fails when git is installed (setuptools_scm issue)
RUN apt-get remove git -y 

RUN apt-get install  -y \
        ros-$ROS_DISTRO-navigation2 \
        ros-$ROS_DISTRO-nav2-bringup \
        ros-$ROS_DISTRO-pointcloud-to-laserscan

WORKDIR /isaac-sim

RUN ./python.sh -m pip install nicegui
RUN apt-get install -y tmux

RUN echo "source /isaac-sim/setup_python_env.sh" >> /root/.bashrc
ENV PYTHONPATH=/isaac-sim:/isaac-sim/roboai:$PYTHONPATH
COPY ./isaac_sim/bin /isaac-sim/roboai/bin
ENV PATH=/isaac-sim/roboai/bin:$PATH



