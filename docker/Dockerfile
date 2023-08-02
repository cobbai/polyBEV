FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

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

# Set environment and working directory
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH
ENV CFLAGS="-I$CUDA_HOME/include $CFLAGS"
ENV FORCE_CUDA="1"
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/miniconda3/envs/bin:$PATH

RUN #conda install --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge cudatoolkit=11.3
RUN #conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ Pillow==8.4.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ tqdm
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ torchpack
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ mmdet==2.20.0 mmsegmentation==0.20.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ nuscenes-devkit
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ mpi4py==3.0.3
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ numba==0.48.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ numpy==1.19.5
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ setuptools==59.5.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ networkx==2.2
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ trimesh==2.35.39
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ torchmetrics==0.3.2

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y openssh-server
RUN apt-get install -y openssh-client
RUN echo root:root | chpasswd
RUN echo "PermitRootLogin yes" >>/etc/ssh/sshd_config

ENV LANG C.UTF-8
CMD ["/usr/sbin/sshd", "-D"]
WORKDIR /home/

# sudo docker run -it -v `pwd`/../bevfusion:/home/bevfusion -p 10001:22 --gpus all --shm-size 16g polybev /bin/bash
