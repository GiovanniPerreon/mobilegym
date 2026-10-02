# All-in-one: llama.cpp server (CUDA) + MobileGym simulator (from your fork) + harness + Chromium.
# Base = official llama.cpp CUDA server image (has llama-server + CUDA libs). Pin a tag if you want reproducibility.
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
FROM ${LLAMA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates git gnupg wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /bench

# Change CACHEBUST (or build with --no-cache) to re-clone after you push to the fork.
ARG REPO_URL=https://github.com/GiovanniPerreon/mobilegym.git
ARG REPO_REF=main
ARG CACHEBUST=1
RUN git clone ${REPO_URL} . && git checkout ${REPO_REF}

RUN npm ci

RUN curl -fL -o mobilegym-data.tar.gz \
      https://github.com/Purewhiter/mobilegym/releases/download/data-v1.0/mobilegym-data-v1.tar.gz \
    && tar -xzf mobilegym-data.tar.gz && rm mobilegym-data.tar.gz

RUN npm run build

RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r bench_env/requirements.txt requests

# Chromium + system libs (shared by Node and Python Playwright)
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN npx playwright install --with-deps chromium && \
    (command -v playwright >/dev/null && playwright install chromium || true) && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LLAMA_CACHE=/cache OUT_DIR=/out
ENTRYPOINT ["/entrypoint.sh"]
