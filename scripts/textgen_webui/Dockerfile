FROM ubuntu:22.04 AS app_base

ARG VERSION_TAG
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV VERSION_TAG=${VERSION_TAG}

COPY sources.list.22.04 /etc/apt/sources.list
RUN apt-get update && apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv python3-pip && \
    pip3 install virtualenv && virtualenv /venv && python3 -m venv $VIRTUAL_ENV && \
    pip3 install --upgrade pip setuptools
RUN git clone https://github.com/oobabooga/text-generation-webui -b ${VERSION_TAG} --depth 1 /app

COPY textgen_webui/build_extensions.sh /scripts/


FROM app_base AS app_nvidia
RUN pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install -r /app/requirements.txt && \
    /scripts/build_extensions.sh

ENTRYPOINT ["python3", "/app/server.py"]


