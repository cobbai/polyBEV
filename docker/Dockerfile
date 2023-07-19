FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Pillow==8.4.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchpack
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nuscenes-devkit
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mpi4py==3.0.3
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numba==0.48.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.19.5
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple setuptools==59.5.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple networkx==2.2
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple trimesh==2.35.39

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y openssh-server
RUN apt-get install -y openssh-client
RUN echo root:root | chpasswd
RUN echo "PermitRootLogin yes" >>/etc/ssh/sshd_config

ENV LANG C.UTF-8
CMD ["/usr/sbin/sshd", "-D"]
WORKDIR /home/

# sudo docker run -it -v `pwd`/../bevfusion:/home/bevfusion -p 10001:22 --gpus all --shm-size 16g bevfusion /bin/bash