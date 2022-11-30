FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
LABEL maintainer="bigscience-workshop"
LABEL repository="petals"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  git \
  ed \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install python~=3.10 pip && \
    pip install --no-cache-dir "torch>=1.12" torchvision torchaudio && \
    conda clean --all && rm -rf ~/.cache/pip

COPY requirements.txt petals/requirements.txt
COPY requirements-dev.txt petals/requirements-dev.txt
RUN pip install --no-cache-dir -r petals/requirements.txt && \
    pip install --no-cache-dir -r petals/requirements-dev.txt

COPY . petals/
RUN pip install -e petals[dev]

WORKDIR /home/petals/
CMD bash