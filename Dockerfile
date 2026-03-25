FROM python:3.11-slim

WORKDIR /app

# Faster & more reliable pip
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System deps (optional): keep minimal. Add build tools only if wheels unavailable.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libexpat1 \
  && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy source (optional). In compose we still bind-mount for fast iteration.
COPY . /app

CMD ["/bin/sh", "-c", "sleep infinity"]
