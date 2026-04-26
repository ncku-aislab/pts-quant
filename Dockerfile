FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    glab \
    wget \
    curl \
    unzip \
    ca-certificates \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install everything except torch/torchvision/torchaudio from requirements if you want stricter control,
# but keeping it simple here:
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=ollama/ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ghcr.io/anomalyco/opencode /usr/bin/opencode /usr/bin/opencode

CMD ["/bin/bash"]
