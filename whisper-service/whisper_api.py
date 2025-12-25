from flask import Flask, request, jsonify
import tempfile
import os
import torch
import whisper
import threading
import time

app = Flask(__name__)

model = whisper.load_model("large", device="cpu") # large, medium, small, tiny

unload_delay = 600
unload_timer = None
lock = threading.Lock()

def schedule_unload():
    global unload_timer
    with lock:
        if unload_timer:
            unload_timer.cancel()
        unload_timer = threading.Timer(unload_delay, unload_model)
        unload_timer.start()

def unload_model():
    global unload_timer
    with lock:
        model.to("cpu")
        torch.cuda.empty_cache()
        unload_timer = None

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
        model.to("cuda")

        result = model.transcribe(tmp_path, language=language)

        schedule_unload()

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(tmp_path)

    return jsonify({
        "text": result["text"],
        "language": result.get("language")
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
