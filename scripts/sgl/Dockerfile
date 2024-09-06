## based on: https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile
ARG CUDA_VERSION=12.4.1
## see check flashinfer version: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server.py
ARG BUILD_TYPE=all

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /sgl-workspace

ARG PYTHON_VERSION=3.10
ARG PYTHON_VERSION_STR=310
ARG SGL_TAG=v0.3.0
ARG FLASHINFER_VER=0.1.6

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') \
    && echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment \
    && echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt update -y \
    && apt install software-properties-common git curl sudo vim python3-pip -y \
    && add-apt-repository ppa:deadsnakes/ppa -y && apt update \
    && apt install python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils -y \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && apt install curl git sudo -y \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py \
    && python3 --version && python3 -m pip --version \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

RUN python3 -m pip install --upgrade pip setuptools wheel html5lib six \
    && git clone --depth=1 https://github.com/sgl-project/sglang.git -b ${SGL_TAG} \
    && cd sglang \
    && if [ "$BUILD_TYPE" = "srt" ]; then \
         python3 -m pip --no-cache-dir install -e "python[srt]"; \
       else \
         python3 -m pip --no-cache-dir install -e "python[all]"; \
       fi

RUN if [ "$CUDA_VERSION" = "12.1.1" ]; then \
        export CUDA_IDENTIFIER=cu121 && \
        python3 -m pip --no-cache-dir install https://github.com/flashinfer-ai/flashinfer/releases/download/v${FLASHINFER_VER}/flashinfer-${FLASHINFER_VER}+cu121torch2.4-cp${PYTHON_VERSION_STR}-cp${PYTHON_VERSION_STR}-linux_x86_64.whl; \
    elif [ "$CUDA_VERSION" = "12.4.1" ]; then \
        export CUDA_IDENTIFIER=cu124 && \
        python3 -m pip --no-cache-dir install https://github.com/flashinfer-ai/flashinfer/releases/download/v${FLASHINFER_VER}/flashinfer-${FLASHINFER_VER}+cu124torch2.4-cp${PYTHON_VERSION_STR}-cp${PYTHON_VERSION_STR}-linux_x86_64.whl; \
    else \
        echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1; \
    fi

RUN python3 -m pip cache purge

ENV DEBIAN_FRONTEND=interactive
