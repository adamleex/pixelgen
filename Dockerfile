FROM 699983977898.dkr.ecr.us-east-1.amazonaws.com/training-base:pt210-cu130-py312

RUN ln -sf /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu && \
    mkdir -p /lib64 && ln -sf /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2 && \
    ldconfig

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --break-system-packages --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ ./src/
COPY configs_t2i/ ./configs_t2i/

ENV PYTHONPATH="/app:${PYTHONPATH:-}"
