from flask import Flask, request, jsonify
import tempfile
import os
import torch
import whisper

app = Flask(__name__)

model = whisper.load_model("medium", device="cpu")

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

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(tmp_path)

        model.to("cpu")
        torch.cuda.empty_cache()

    return jsonify({
        "text": result["text"],
        "language": result.get("language")
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
