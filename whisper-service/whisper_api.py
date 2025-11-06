from flask import Flask, request, jsonify
import tempfile
import os
import whisper

app = Flask(__name__)

model = whisper.load_model("large", device="cuda")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    language = request.form.get("language")  # например "en"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if language:
            result = model.transcribe(tmp_path, language=language)
        else:
            result = model.transcribe(tmp_path)  # автоопределение языка

        return jsonify({
            "text": result["text"],
            "language": result.get("language")  # Whisper возвращает определённый язык
        })
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
