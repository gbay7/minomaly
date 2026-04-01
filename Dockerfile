FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

# PyG and deps
RUN pip install --no-cache-dir \
    torch-geometric==2.6.1 \
    torch-scatter torch-sparse torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Python deps
RUN pip install --no-cache-dir \
    networkx \
    scikit-learn \
    scipy \
    matplotlib \
    tqdm \
    pyyaml \
    dacite \
    requests

# Copy package
COPY minomaly/ /app/minomaly/
COPY configs/ /app/configs/

# Pre-create output dirs
RUN mkdir -p /app/ckpt /app/plots /app/results

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["python", "-m", "minomaly"]
CMD ["--config", "configs/default.yaml"]
