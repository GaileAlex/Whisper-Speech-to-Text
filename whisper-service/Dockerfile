FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip

# CUDA PyTorch
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu128

# faster whisper + gunicorn
RUN pip3 install faster-whisper flask gunicorn

RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu')"

COPY whisper_api.py .

EXPOSE 5001

CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5001", "whisper_api:app"]
