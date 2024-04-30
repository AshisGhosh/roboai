FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx

# RUN apt-get install -y python3 python3-dev python3-distutils
# RUN apt-get install -y python-is-python3
# RUN wget https://bootstrap.pypa.io/get-pip.py
# RUN python3 get-pip.py

RUN git clone https://github.com/mahaoxiang822/Scale-Balanced-Grasp.git

# RUN cd Scale-Balanced-Grasp && pip install -r requirements.txt
RUN pip install \
    torch \
    tensorboard \
    numpy \
    scipy \
    open3d \
    Pillow \
    tqdm

# RUN cd Scale-Balanced-Grasp/pointnet2 && python setup.py install
# RUN cd Scale-Balanced-Grasp/knn && python setup.py install

# RUN git clone https://github.com/graspnet/graspnetAPI.git
# RUN cd graspnetAPI && pip install .