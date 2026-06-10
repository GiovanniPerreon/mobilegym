# ---- Builder stage ----
FROM node:22-slim AS builder

# Install Python 3.11 and required system packages
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv \
    curl git wget \
    libglib2.0-0 libnss3 libnspr4 libatk-bridge2.0-0 \
    libdrm2 libxkbcommon0 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Clone your fork (replace with your actual branch if needed)
RUN git clone https://github.com/GiovanniPerreon/mobilegym.git . \
    && git checkout main   # or master, or a specific commit

# Install Node dependencies
RUN npm ci

# Download and extract the dataset (required for many benchmark tasks)
RUN curl -L -o mobilegym-data.tar.gz \
      https://github.com/Purewhiter/mobilegym/releases/download/data-v1.0/mobilegym-data-v1.tar.gz \
    && tar -xzf mobilegym-data.tar.gz \
    && rm mobilegym-data.tar.gz

# Build the frontend (produces dist/)
RUN npm run build

# Install Python benchmark dependencies and Playwright
RUN pip install --no-cache-dir -r bench_env/requirements.txt
RUN npx playwright install chromium

# ---- Final stage (smaller runtime image) ----
FROM node:22-slim

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip \
    libglib2.0-0 libnss3 libnspr4 libatk-bridge2.0-0 \
    libdrm2 libxkbcommon0 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy built frontend, benchmark code, Python packages, Playwright browsers, and dataset from builder
COPY --from=builder /app/dist /app/dist
COPY --from=builder /app/bench_env /app/bench_env
COPY --from=builder /app/mobilegym-data /app/mobilegym-data   # <-- dataset included
COPY --from=builder /app/package*.json /app/
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

# Expose the preview server port
EXPOSE 4173

# Default command: start the preview server (ready for benchmark)
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]