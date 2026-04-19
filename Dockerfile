FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG CPU_TORCH_ONLY=1

WORKDIR /app

# Minimal native deps for geospatial/runtime wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv>=0.5.0"

# --- Dependencies layer (cached unless pyproject.toml/uv.lock/sam3 change) ---
COPY pyproject.toml uv.lock ./
COPY vendor/sam3 ./vendor/sam3

RUN if [ "${CPU_TORCH_ONLY}" = "1" ]; then \
        uv sync --frozen --no-dev --no-install-package torch --no-install-package torchvision \
        && uv pip install --index-url ${TORCH_INDEX_URL} torch torchvision; \
    else \
        uv sync --frozen --no-dev; \
    fi

ENV PATH="/app/.venv/bin:${PATH}"

# --- Application layer (rebuilds in seconds on code changes) ---
COPY hitl ./hitl
COPY config ./config
COPY scripts ./scripts

EXPOSE 8000

CMD ["python", "-m", "hitl.app"]
