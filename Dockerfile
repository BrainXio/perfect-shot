# Updated Dockerfile (add libGL for OpenCV)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ src/
COPY __init__.py .
COPY sample.jpg .
COPY entrypoint.sh .
ENV PATH=/root/.local/bin:$PATH

RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["sample.jpg", "output.jpg"]