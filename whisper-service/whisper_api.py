from flask import Flask, request, jsonify
import tempfile
import os
import time
import torch
from faster_whisper import WhisperModel

app = Flask(__name__)

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

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
        start_time = time.time()

        torch.cuda.empty_cache()

        print("🔄 Loading Whisper to GPU...")
        model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )

        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=1
        )

        text = "".join([seg.text for seg in segments])

        del model
        torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        print(f"🧠 Done in {elapsed:.2f}s")

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
