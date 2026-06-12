# ---- Builder stage ----
FROM node:22-slim AS builder

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv \
    curl git wget \
    libglib2.0-0 libnss3 libnspr4 libatk-bridge2.0-0 \
    libdrm2 libxkbcommon0 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN git clone https://github.com/GiovanniPerreon/mobilegym.git . \
    && git checkout main

RUN npm ci

RUN curl -L -o mobilegym-data.tar.gz \
      https://github.com/Purewhiter/mobilegym/releases/download/data-v1.0/mobilegym-data-v1.tar.gz \
    && tar -xzf mobilegym-data.tar.gz \
    && rm mobilegym-data.tar.gz

RUN npm run build

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir -r bench_env/requirements.txt
RUN pip install requests

RUN npx playwright install chromium

# ---- Final stage ----
FROM node:22-slim

RUN apt-get update && apt-get install -y \
    python3.11 \
    libglib2.0-0 libnss3 libnspr4 libatk-bridge2.0-0 \
    libdrm2 libxkbcommon0 libgbm1 libasound2 \
    libcups2 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY --from=builder /app/package*.json ./
RUN npm ci

COPY --from=builder /app/dist /app/dist
COPY --from=builder /app/bench_env /app/bench_env
COPY --from=builder /app/mobilegym-data /app/mobilegym-data
COPY --from=builder /app/apps /app/apps
COPY --from=builder /app/system /app/system
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

EXPOSE 4173
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]