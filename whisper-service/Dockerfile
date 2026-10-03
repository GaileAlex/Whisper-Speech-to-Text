# cuBLAS 12 + cuDNN 9 (required by ctranslate2 4.5+; CUDA 12.8+ is needed for Blackwell GPUs).
# Not CUDA 13: the ctranslate2 wheels link the CUDA 12 libraries (libcublas.so.12).
# Ubuntu 24.04 has Python 3.12; the 3.10 of Ubuntu 22.04 is end-of-life and gets no new numpy/onnxruntime/av.
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

# tzdata: log timestamps follow TZ from docker-compose.yml
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv ffmpeg tzdata && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# The system Python of Ubuntu 24.04 refuses pip installs (PEP 668): the packages go to a virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
RUN python3 -m venv $VIRTUAL_ENV && pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY whisper_api.py .

# Uploaded media is parsed by ffmpeg, so don't run as root.
# The model cache lives in ~/.cache/huggingface, the CUDA kernel cache in ~/.nv (both mounted in docker-compose.yml).
# ~/.nv is created here so that the named volume mounted over it is owned by whisper, not root; ~/.cache too, or
# Docker creates it for the mount as root and onnxruntime (VAD) cannot keep its files there.
# The user ubuntu of the base image is removed so that whisper keeps uid 1000, the owner of the volumes created with
# the former Ubuntu 22.04 image.
RUN userdel --remove ubuntu && useradd --create-home --uid 1000 whisper && \
    install -d -o whisper -g whisper /home/whisper/.nv /home/whisper/.cache
USER whisper

EXPOSE 5001

CMD ["gunicorn", "-w", "1", "--threads", "2", "--timeout", "6000", "--graceful-timeout", "600", "-b", "0.0.0.0:5001", "whisper_api:app"]
