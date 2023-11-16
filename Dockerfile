FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG NO_RECS="--no-install-recommends"
RUN apt-get update -y \ 
    && apt-get install $NO_RECS -y \
        wget=1.20.3-1ubuntu2 \
        gcc-10 \
        g++-10 \
        git \
        python3-dev \
        python3-pip \
    # Clear lists.
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Install Bazelisk.
    && wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64 \
    && chmod +x bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/local/bin/bazel \

ARG JAX_PACKAGE_URL="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
ARG NO_CACHE="--no-cache-dir"
RUN python3 -m pip install $NO_CACHE --upgrade pip \
    && python3 -m pip install $NO_CACHE \
        "jax[cuda11_cudnn86]" -f $JAX_PACKAGE_URL \
    && python3 -m pip install $NO_CACHE \
        jupyterlab==4.0.5

WORKDIR project
EXPOSE 7070
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]