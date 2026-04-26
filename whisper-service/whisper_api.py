from flask import Flask, request, jsonify
import tempfile
import os
import time
import threading
import subprocess
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


# ----------------------------
# Model management
# ----------------------------
def load_model():
    global model
    if model is None:
        print("🚀 Loading Whisper model...")
        model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("✅ Model loaded")


def unload_model():
    global model
    if model is not None:
        print("🧹 Unloading model...")
        del model
        model = None
        torch.cuda.empty_cache()
        print("✅ GPU memory cleared")


def watchdog():
    global last_used
    while True:
        time.sleep(60)
        if model is not None and time.time() - last_used > IDLE_TIMEOUT:
            unload_model()


threading.Thread(target=watchdog, daemon=True).start()


# ----------------------------
# Audio normalization (CRITICAL FIX)
# ----------------------------
def convert_to_wav(input_path):
    output_path = input_path + ".wav"

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path


# ----------------------------
# Endpoint
# ----------------------------
@app.route("/transcribe", methods=["POST"])
def transcribe():
    global last_used

    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    language = request.form.get("language")

    tmp_path = None
    wav_path = None

    try:
        # Save upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Convert to clean audio
        wav_path = convert_to_wav(tmp_path)

        with lock:
            load_model()
            last_used = time.time()

            start_time = time.time()

            segments, info = model.transcribe(
                wav_path,
                language=language,
                task="transcribe",
                beam_size=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=700
                ),
                condition_on_previous_text=False
            )

            segments = list(segments)

        # Clean output text (avoid repetition artifacts)
        text = " ".join(seg.text.strip() for seg in segments)

        elapsed = time.time() - start_time
        print(f"🧠 Done in {elapsed:.2f}s")

        return jsonify({
            "text": text,
            "language": info.language,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip()
                }
                for s in segments
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # cleanup
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p):
                os.remove(p)


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
