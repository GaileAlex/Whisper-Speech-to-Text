# CUDA 11.8 runtime с Ubuntu 22.04
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Устанавливаем Python и утилиты
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv \
    ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Обновляем pip
RUN pip3 install --upgrade pip

# Ставим PyTorch с поддержкой CUDA 11.8
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Ставим Whisper и Flask
RUN pip3 install git+https://github.com/openai/whisper.git flask

# Копируем Flask-приложение
COPY whisper_api.py .

# Открываем порт
EXPOSE 5001

# Запуск приложения
CMD ["python3", "whisper_api.py"]
