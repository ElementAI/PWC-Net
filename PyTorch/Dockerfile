FROM nvidia/cuda:10.0-devel-ubuntu18.04

# Install native dependences
RUN apt-get update && apt-get install -y curl \
    build-essential git libreadline-dev zlib1g-dev libssl-dev \
    libbz2-dev libsqlite3-dev libffi-dev libglib2.0-0

# Install pyenv
WORKDIR /srv/app
ENV PYENV_ROOT /srv/app/.pyenv
RUN git clone git://github.com/yyuu/pyenv.git $PYENV_ROOT
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install python
RUN pyenv install 3.7.3 
RUN pyenv global 3.7.3
ENV PATH $PYENV_ROOT/versions/3.7.3/bin:$PATH
RUN pip install --upgrade pip

# CUDA 10.0
RUN pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install visdom dominate opencv-python-headless

COPY ./external_packages /external_packages
ENV FORCE_CUDA TRUE
WORKDIR /external_packages/correlation-pytorch-master/correlation-pytorch
RUN python setup.py build install
WORKDIR /app

RUN pip install pytest
