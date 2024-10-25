# Use the base image with PyTorch and CUDA support
# FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive

# tzdata
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone && \
    apt-get clean

RUN apt install -y git vim wget unzip

RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libeigen3-dev freeglut3-dev libsm6 libxext6 libxrender-dev

COPY ./entrypoint.sh /
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]