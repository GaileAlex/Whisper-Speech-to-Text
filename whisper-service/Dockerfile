FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

# PyTorch под CUDA 12.3
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# faster-whisper + сервер
RUN pip install faster-whisper flask gunicorn

COPY whisper_api.py .

EXPOSE 5001

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5001", "whisper_api:app"]
