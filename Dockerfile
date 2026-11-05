# ═══════════════════════════════════════════════════════════
#  WeatherProj — Hybrid Physics-AI Smog Forecaster
#  Dockerfile
# ═══════════════════════════════════════════════════════════
#
#  Build:
#    docker build -t weatherproj .
#
#  Run — Manual input mode (interactive):
#    docker run -it weatherproj
#
#  Run — CSV mode (mount your CSV folder):
#    docker run -it -v "C:/MyData:/data" -e CSV_INPUT_PATH=/data/inputs.csv weatherproj
#
#  NOTE: Python 3.14 (used locally) has no official Docker image yet.
#  Python 3.12-slim is fully compatible with all packages in this project.
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Builder ───────────────────────────────────────
FROM python:3.12-slim AS builder

# Keeps Python from buffering stdout/stderr (see logs immediately)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build tools needed by some JAX/jaxlib wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ───────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # JAX: use 64-bit precision (required by this project)
    JAX_ENABLE_X64=true \
    # Default CSV path inside the container
    # Override at runtime: -e CSV_INPUT_PATH=/data/your_file.csv
    CSV_INPUT_PATH=/app/inputs.csv

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy only the application source files (not __pycache__, etc.)
COPY main.py           .
COPY ai_model.py       .
COPY mechanism_core.py .
COPY solver_module.py  .
COPY trainer.py        .
COPY inputs.csv        .

# Create a /data mount point for external CSV files
# Usage: docker run -v "/host/path:/data" -e CSV_INPUT_PATH=/data/file.csv
RUN mkdir /data

# Non-root user for security best practices
RUN useradd --create-home appuser && chown -R appuser /app /data
USER appuser

# The app is interactive — it reads stdin for mode selection and manual inputs.
# Always run with:  docker run -it weatherproj
CMD ["python", "main.py"]
