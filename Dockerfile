FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
LABEL maintainer="meta-introspector"
LABEL repository="petals"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  git \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install python~=3.10.12 pip && \
    pip install --no-cache-dir "torch>=1.12" && \
    conda clean --all && rm -rf ~/.cache/pip

VOLUME /cache
ENV PETALS_CACHE=/cache

ADD pip.freeze petals/pip.freeze
RUN pip install --no-cache-dir -r petals/pip.freeze
ADD pip2.freeze petals/pip2.freeze
RUN pip install --no-cache-dir -r petals/pip2.freeze

ADD tests petals/tests
ADD src petals/src
ADD LICENSE  README.md pyproject.toml setup.cfg petals/

RUN pip install --no-cache-dir -e petals

RUN pip freeze > pip.freeze.new
#RUN pip install --no-cache-dir --upgrade transformers==4.34.0

WORKDIR /home/petals/

RUN pip freeze > pip.freeze.new

CMD python -m petals.cli.run_server --port 31331  --num_blocks=1 Maykeye/TinyLLama-v0 