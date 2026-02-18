FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv \
    ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install git+https://github.com/openai/whisper.git flask

COPY whisper_api.py .

EXPOSE 5001

CMD ["python3", "whisper_api.py"]
