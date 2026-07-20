"""
app.py
------
Classroom Attendance Counter — upload a photo of the class, get an
instant headcount with faces marked, instead of a manual roll call.
"""

import base64
import logging
import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from face_detector import FaceDetector

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload cap

detector = FaceDetector()
print(f"[app] Starting on {detector.backend.upper()} backend (may upgrade to DNN in the background).")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", backend=detector.backend)


@app.route("/detect", methods=["POST"])
def detect():
    if "photo" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG, PNG, BMP, or WEBP image."}), 400

    try:
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not read that image. The file may be corrupted."}), 400

        start = time.time()
        boxes, image, backend_used = detector.detect(image)
        elapsed_ms = int((time.time() - start) * 1000)

        annotated = draw_boxes(image, boxes)
        encoded_image = encode_image_to_base64(annotated)

        return jsonify(
            {
                "count": len(boxes),
                "image": encoded_image,
                "backend": backend_used,
                "elapsed_ms": elapsed_ms,
            }
        )
    except Exception:
        # Never leak a stack trace to the client — log it server-side and
        # return a clean, generic error instead.
        logger.exception("Unhandled error while processing an uploaded image")
        return jsonify({"error": "Something went wrong while analyzing that photo. Please try a different image."}), 500


@app.errorhandler(413)
def file_too_large(_error):
    max_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify({"error": f"That photo is too large. Please upload something under {max_mb} MB."}), 413


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({"error": "Something went wrong on the server. Please try again."}), 500


@app.route("/favicon.ico")
def favicon():
    return "", 204


def draw_boxes(image: np.ndarray, boxes) -> np.ndarray:
    annotated = image.copy()
    box_color = (60, 179, 217)      # warm mustard-ink accent, BGR
    text_bg_color = (34, 42, 27)    # near-black ink, BGR

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        label = str(idx)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x, y - th - 10), (x + tw + 8, y), text_bg_color, -1)
        cv2.putText(annotated, label, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated


def encode_image_to_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Failed to encode result image.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


if __name__ == "__main__":
    # Debug mode is OFF by default: Flask's interactive debugger allows
    # arbitrary code execution and must never be enabled outside of local
    # development. Opt in explicitly with `FLASK_DEBUG=1 python app.py`.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"

    # Binds to localhost only by default. Set HOST=0.0.0.0 if you want other
    # devices on the same classroom Wi-Fi (e.g. a phone) to reach it too.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))

    app.run(debug=debug_mode, host=host, port=port)
