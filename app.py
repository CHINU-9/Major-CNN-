from flask import Flask, render_template, request, send_file
import os
from detection import extract_features, detect_accident, save_clip

app = Flask(__name__)

os.makedirs("uploads", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]
    video_path = os.path.join("uploads", file.filename)
    file.save(video_path)

    features = extract_features(video_path)
    accident_frames = detect_accident(features)

    if len(accident_frames) > 0:
        output_file = save_clip(video_path, accident_frames[0])
    else:
        output_file = save_clip(video_path, len(features)//2)

    return send_file(output_file, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
