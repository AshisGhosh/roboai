FROM nvcr.io/nvidia/isaac-sim:2023.1.1

RUN ./python.sh -m pip install \
        fastapi \
        uvicorn \
        python-multipart

COPY ./isaac_sim/omniverse_patch/nucleus.py /isaac-sim/exts/omni.isaac.core/omni/isaac/core/utils/nucleus.py

RUN apt-get update && apt-get install -y \
                 libglib2.0-0 \
                 libsm6 \
                 libxrender1 \
                 libxext6

RUN ./python.sh -m pip install \
        opencv-python