# =============================================================================
# Chonkie OSS API – Dockerfile
# Multi-stage build: builder stage installs dependencies, final stage is lean.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 – builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and src/ for installation
COPY pyproject.toml .
COPY src/ ./src/

# Create a virtual environment and install Chonkie with API dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .[api,semantic,code,openai]

# ---------------------------------------------------------------------------
# Stage 2 – runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only runtime system libraries (tree-sitter requires no extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source
COPY api/ ./api/

# Non-root user for security
RUN useradd --create-home --shell /bin/bash chonkie
USER chonkie

# Expose the default API port
EXPOSE 8000

# Health check so container orchestrators can verify the service is up
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ---------------------------------------------------------------------------
# Runtime configuration (override with -e / docker-compose environment:)
# ---------------------------------------------------------------------------
ENV LOG_LEVEL="INFO"
ENV CORS_ORIGINS="*"
# OPENAI_API_KEY – required only if you use /v1/refine/embeddings

# ---------------------------------------------------------------------------
# Start the server
# ---------------------------------------------------------------------------
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
