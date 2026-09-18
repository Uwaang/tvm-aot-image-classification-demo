FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TVM_HOME=/opt/tvm \
    PYTHONPATH=/opt/tvm/python \
    LD_LIBRARY_PATH=/opt/tvm/build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    ninja-build \
    libtinfo-dev \
    zlib1g-dev \
    libxml2-dev \
    libedit-dev \
    llvm-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# The old Relay + AOT/CRT path is no longer shipped by current PyPI wheels.
# Build the official TVM v0.14.0 source instead, with microTVM enabled.
RUN git clone --branch v0.14.0 --depth 1 --recursive --shallow-submodules \
        https://github.com/apache/tvm.git /opt/tvm \
    && mkdir -p /opt/tvm/build \
    && cp /opt/tvm/cmake/config.cmake /opt/tvm/build/config.cmake \
    && printf '\nset(CMAKE_BUILD_TYPE Release)\nset(USE_LLVM llvm-config)\nset(USE_MICRO ON)\n' \
        >> /opt/tvm/build/config.cmake \
    && cmake -S /opt/tvm -B /opt/tvm/build -G Ninja \
    && cmake --build /opt/tvm/build --parallel 2 \
    && pip install -e /opt/tvm/python \
    && pip install --force-reinstall "numpy==1.26.4"

COPY . .

CMD ["bash", "scripts/run_demo.sh"]
