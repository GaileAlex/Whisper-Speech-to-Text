from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os
import torch

app = Flask(__name__)

MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "float16"

# Загружаем модель сразу при старте контейнера
print("Starting container, loading model...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE_TYPE)
print("Model ready!")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    language = request.form.get("language")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5
        )

        text = " ".join([segment.text for segment in segments])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(tmp_path)

    return jsonify({
        "text": text,
        "language": info.language
    })
