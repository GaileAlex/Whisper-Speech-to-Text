from flask import Flask, request, jsonify
import tempfile
import os
import time
import threading
import torch
from faster_whisper import WhisperModel

app = Flask(__name__)

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"

IDLE_TIMEOUT = 30 * 60

model = None
last_used = time.time()
lock = threading.Lock()

print("🚀 Whisper service starting...")


def load_model():
    global model
    if model is None:
        print("🚀 Loading Whisper model into GPU...")
        model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("✅ Model loaded")


def unload_model():
    global model
    if model is not None:
        print("🧹 Unloading Whisper model from GPU...")
        del model
        model = None
        torch.cuda.empty_cache()
        print("✅ GPU memory freed")


def watchdog():
    global last_used
    while True:
        time.sleep(60)
        if model is not None:
            if time.time() - last_used > IDLE_TIMEOUT:
                unload_model()


threading.Thread(target=watchdog, daemon=True).start()


@app.route("/transcribe", methods=["POST"])
def transcribe():
    global last_used

    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    language = request.form.get("language")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with lock:
            load_model()
            last_used = time.time()

            start_time = time.time()

            segments, info = model.transcribe(
                tmp_path,
                language=language,
                beam_size=1
            )

        text = "".join([seg.text for seg in segments])

        elapsed = time.time() - start_time
        print(f"🧠 Done in {elapsed:.2f}s")

        return jsonify({
            "text": text,
            "language": info.language
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
