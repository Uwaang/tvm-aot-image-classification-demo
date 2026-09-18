FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
COPY scripts/install_tvm_wheel.py scripts/install_tvm_wheel.py

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python scripts/install_tvm_wheel.py

COPY . .

CMD ["bash", "scripts/run_demo.sh"]
