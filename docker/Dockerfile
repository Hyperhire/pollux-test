# Use the base image with PyTorch and CUDA support
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update

RUN apt install -y git vim wget unzip

RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0

COPY ./entrypoint.sh /
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]