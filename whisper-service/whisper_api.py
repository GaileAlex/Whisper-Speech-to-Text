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
COMPUTE_TYPE = "float16"

model = None
model_lock = threading.Lock()
last_used = time.time()

UNLOAD_DELAY = 120  # секунд

# --- GPU HELPERS ---

def gpu_memory_used():
    return torch.cuda.memory_allocated() / 1024**3

def gpu_memory_total():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3

def gpu_is_busy():
    return gpu_memory_used() / gpu_memory_total() > 0.8


# --- MODEL MANAGEMENT ---

def load_model():
    global model
    print("🔄 Loading Whisper to GPU...")
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)


def unload_model():
    global model
    with model_lock:
        if model is not None:
            print("🧹 Unloading Whisper from GPU...")
            del model
            model = None
            torch.cuda.empty_cache()


def ensure_model():
    global model, last_used

    with model_lock:
        if model is None:
            load_model()

        last_used = time.time()


# --- AUTO UNLOAD THREAD ---

def auto_unload_worker():
    global last_used
    while True:
        time.sleep(10)
        if model is None:
            continue

        idle_time = time.time() - last_used

        if idle_time > UNLOAD_DELAY or gpu_is_busy():
            unload_model()


threading.Thread(target=auto_unload_worker, daemon=True).start()


# --- API ---

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
        ensure_model()

        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=1
        )

        text = "".join([seg.text for seg in segments])

        last_used = time.time()

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(tmp_path)

    return jsonify({
        "text": text,
        "language": info.language
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
