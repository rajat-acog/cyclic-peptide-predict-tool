# Lean Boltz-2 image for cyclic-peptide-predict (stock boltz 2.2.1, MIT licence).
# Weights are NOT in the image: they are mounted from the shared cache at run time.
# Our Python (cyclic_tool) is mounted at /scripts at run time, so changing it never needs a rebuild.
FROM python:3.11-slim
ARG BOLTZ_VERSION=2.2.1
# build-essential: Triton / cuEquivariance compile small helpers on first use.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --retries 10 --timeout 120 "boltz[cuda]==${BOLTZ_VERSION}" \
    && boltz predict --help >/dev/null
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
