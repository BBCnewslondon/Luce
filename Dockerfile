# Stage 1: build TA-Lib C library and compile Python wheels
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library from source
RUN wget -q https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && cd / \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Build Python wheels so the runtime stage needs no compiler
COPY requirements.txt /tmp/requirements.txt
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r /tmp/requirements.txt

# Stage 2: lean runtime image
FROM python:3.11-slim AS runtime

# Copy compiled TA-Lib shared libraries
COPY --from=builder /usr/local/lib/libta_lib* /usr/local/lib/
COPY --from=builder /usr/local/include/ta-lib /usr/local/include/ta-lib

RUN ldconfig

WORKDIR /app

# Install pre-built wheels — no compiler required at this stage
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application source
COPY . .

# Create runtime directories for data, models, and logs
RUN mkdir -p data/features data/models data/metrics logs

# Environment variables — override at runtime via .env or Azure App Settings.
# OANDA_ACCOUNT_ID and OANDA_API_TOKEN must be provided; the container will
# fail with a clear error if they are absent (enforced by the application).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OANDA_ENVIRONMENT="practice"

# Expose a port for any future HTTP health-check endpoint
EXPOSE 8080

# Default entry point runs the data ingestion pipeline.
# Override at runtime to execute other pipeline stages, for example:
#   docker run luce python -m signal_generation.ensemble
#   docker run luce python -m execution.order_executor
CMD ["python", "-m", "data_ingestion.pipeline"]
