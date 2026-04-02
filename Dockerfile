# training-base: NGC 25.11 (PyTorch 2.10 + CUDA 13.0 + Python 3.12) + EFA 1.43.2
FROM 699983977898.dkr.ecr.us-east-1.amazonaws.com/ds-core-cn-omra-trainer@sha256:068cef8d9ae6aefb21eb07a4a3dc8c0020be73c0e0cd55b9b770952dc5a255d3

RUN ln -sf /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu && \
    mkdir -p /lib64 && ln -sf /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2 && \
    ldconfig

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --break-system-packages --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ ./src/
COPY configs_t2i/ ./configs_t2i/

ENV PYTHONPATH="/app"
