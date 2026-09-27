# cuBLAS 12.8 + cuDNN 9 (required by ctranslate2 4.5+; CUDA 12.8 is needed for Blackwell GPUs)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# tzdata: log timestamps follow TZ from docker-compose.yml
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg tzdata && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# The system pip of Ubuntu 22.04 fails to resolve these requirements
RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY whisper_api.py .

# Uploaded media is parsed by ffmpeg, so don't run as root.
# The model cache lives in ~/.cache/huggingface, the CUDA kernel cache in ~/.nv (both mounted in docker-compose.yml).
# ~/.nv is created here so that the named volume mounted over it is owned by whisper, not root.
RUN useradd --create-home whisper && install -d -o whisper -g whisper /home/whisper/.nv
USER whisper

EXPOSE 5001

CMD ["gunicorn", "-w", "1", "--threads", "2", "--timeout", "6000", "--graceful-timeout", "600", "-b", "0.0.0.0:5001", "whisper_api:app"]
