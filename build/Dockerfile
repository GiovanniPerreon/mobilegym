# mobilegym_full: llama.cpp server (CUDA) + MobileGym simulator + bench_env harness + Chromium, in one image.
# Rebuilds the image used for every run so far (built on 2 Oct 2026), with every version fixed:
#   llama.cpp b11347 (revision 5fc4f3c8), MobileGym commit e4307c1, data release v0.1.0,
#   Python packages from requirements.lock.txt (read from that image's /venv).
# Build it with the same name on every node (cluster guide, section 7), from the repository folder:
#   docker build -f build/Dockerfile -t mobilegym_full:latest build/
FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b11347

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates git gnupg wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /bench
ARG REPO_URL=https://github.com/GiovanniPerreon/mobilegym.git
ARG REPO_REF=e4307c1011e6501a2253f4756f69601581abf67c
RUN git clone ${REPO_URL} . && git checkout ${REPO_REF}
RUN npm ci

ARG DATA_URL=https://github.com/Purewhiter/mobilegym/releases/download/data-v0.1.0/mobilegym-data-v0.1.0.tar.gz
RUN curl -fL -o mobilegym-data.tar.gz ${DATA_URL} && tar -xzf mobilegym-data.tar.gz && rm mobilegym-data.tar.gz
RUN npm run build

RUN python3.11 -m venv /venv
ENV PATH=/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir pip==26.2.1 && pip install --no-cache-dir -r /tmp/requirements.lock.txt

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN npx playwright install --with-deps chromium && \
    (command -v playwright >/dev/null && playwright install chromium || true) && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.py /entrypoint.py
ENV LLAMA_CACHE=/cache OUT_DIR=/out
ENTRYPOINT ["python", "/entrypoint.py"]
