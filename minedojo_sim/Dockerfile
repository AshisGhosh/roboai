FROM ubuntu:22.04

RUN apt-get update -y && apt-get install -y \
                software-properties-common 
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get update -y && apt-get install -y openjdk-8-jdk
RUN update-alternatives --config java

RUN apt-get install -y \
     xvfb  xserver-xephyr tigervnc-standalone-server python3-opengl ffmpeg

RUN apt-get install -y git \
                python3-pip \
                python-is-python3 

WORKDIR /root
RUN git clone https://github.com/thomashopkins32/MineDojo.git
# RUN git clone https://github.com/MineDojo/MineDojo
RUN git clone https://github.com/verityw/MixinGradle-dcfaf61.git MineDojo/minedojo/sim/Malmo/Minecraft/MixinGradle-dcfaf61
COPY patches/build.gradle /root/MineDojo/minedojo/sim/Malmo/Minecraft/build.gradle

# RUN apt-get install -y wget
# RUN wget https://github.com/microsoft/malmo/archive/refs/tags/0.37.0.tar.gz
# RUN tar -xvf 0.37.0.tar.gz
# RUN rm -rf MineDojo/minedojo/sim/Malmo
# RUN mv malmo-0.37.0 MineDojo/minedojo/sim/Malmo
# # install cmake, boost, doxygen, swig
# RUN apt-get install -y cmake \
#                 libboost-all-dev \
#                 doxygen \
#                 swig

# RUN mkdir -p /root/MineDojo/minedojo/sim/Malmo/build
# WORKDIR /root/MineDojo/minedojo/sim/Malmo/build
# RUN cmake -DBoost_INCLUDE_DIR=/home/$USER/boost/boost_1_66_0/include -DCMAKE_BUILD_TYPE=Release ..
# RUN make -j20
RUN cd MineDojo && pip install -e .
