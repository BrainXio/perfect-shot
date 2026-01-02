# Stage 1: Builder
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY test.py .

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "test.py"]