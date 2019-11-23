FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y libglib2.0-0 python-pip
RUN pip install --upgrade pip

RUN pip install numpy==1.14 http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
RUN pip install scipy==1.2.2 torchvision==0.2.0 visdom==0.1.8 dominate==2.4.0 opencv-python-headless==4.1.2.30 cffi==1.13.2

COPY ./external_packages /external_packages
ENV FORCE_CUDA TRUE
WORKDIR /external_packages/correlation-pytorch-master/
RUN bash ./make_cuda.sh
WORKDIR /app
