"""
face_detector.py
-----------------
Face detection engine for the classroom attendance counter.

Two backends are supported:
  1. DNN  - OpenCV's res10 SSD Caffe face detector. Much more accurate,
            handles angled faces, varied lighting, partial occlusion, and
            small/far-away faces (all common in a classroom photo).
  2. Haar - Classic Viola-Jones cascade. Zero download required, since it
            ships inside the opencv-python package.

The app starts up on Haar immediately — no network call blocks startup.
In the background, a thread tries to download the DNN model files
(~10 MB, one-time, from OpenCV's own GitHub repo) and, if that succeeds,
swaps the active backend over to DNN for every request from then on. If
the download fails or is slow, the app just keeps serving from Haar —
nothing ever hangs or breaks.
"""

import os
import socket
import threading
import urllib.request
import cv2
import numpy as np

DOWNLOAD_TIMEOUT_SECONDS = 20

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Official OpenCV repo — same files used inside opencv/samples/dnn/face_detector
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

DNN_CONFIDENCE_THRESHOLD = 0.6
HAAR_SCALE_FACTOR = 1.08
HAAR_MIN_NEIGHBORS = 7
HAAR_MIN_SIZE = (30, 30)

MAX_DIMENSION = 1600  # downscale huge phone photos before processing


class FaceDetector:
    def __init__(self):
        self._lock = threading.Lock()
        self.backend = None
        self.net = None
        self.haar = None

        # Haar loads instantly (no network) so the app is ready to serve
        # requests right away, even before the DNN upgrade attempt finishes.
        self._load_haar()

        os.makedirs(MODELS_DIR, exist_ok=True)
        threading.Thread(target=self._upgrade_to_dnn_in_background, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Backend setup
    # ------------------------------------------------------------------ #
    def _load_haar(self):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        haar = cv2.CascadeClassifier(cascade_path)
        if haar.empty():
            raise RuntimeError("Could not load Haar cascade classifier — OpenCV install looks broken.")
        with self._lock:
            self.haar = haar
            self.backend = "haar"

    def _upgrade_to_dnn_in_background(self):
        if not self._ensure_dnn_files():
            print("[face_detector] DNN model unavailable — staying on Haar cascade.")
            return
        try:
            net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        except Exception as e:
            print(f"[face_detector] Downloaded DNN files but failed to load them, staying on Haar: {e}")
            return

        with self._lock:
            self.net = net
            self.backend = "dnn"
        print("[face_detector] Upgraded to DNN backend.")

    def _ensure_dnn_files(self) -> bool:
        """Return True if both DNN model files are present locally (downloading if needed)."""
        try:
            self._download_if_missing(PROTOTXT_URL, PROTOTXT_PATH)
            self._download_if_missing(CAFFEMODEL_URL, CAFFEMODEL_PATH)
            return os.path.exists(PROTOTXT_PATH) and os.path.exists(CAFFEMODEL_PATH)
        except Exception as e:
            print(f"[face_detector] Could not download DNN model files ({e}).")
            return False

    @staticmethod
    def _download_if_missing(url: str, dest_path: str) -> None:
        if os.path.exists(dest_path):
            return
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(DOWNLOAD_TIMEOUT_SECONDS)
        try:
            urllib.request.urlretrieve(url, dest_path)
        except Exception:
            # Don't leave a truncated/corrupt file behind — that would make
            # os.path.exists() lie to us on the next run and permanently
            # skip retrying the download.
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise
        finally:
            socket.setdefaulttimeout(old_timeout)

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def detect(self, image: np.ndarray):
        """
        Detect faces in a BGR OpenCV image.
        Returns (list_of_boxes, resized_image, backend_name).
        """
        image = self._resize_if_needed(image)

        # Snapshot the active backend under the lock so a mid-request
        # upgrade from Haar to DNN (happening on the background thread)
        # can never leave us reading a half-swapped state.
        with self._lock:
            backend, net, haar = self.backend, self.net, self.haar

        if backend == "dnn":
            boxes = self._detect_dnn(image, net)
        else:
            boxes = self._detect_haar(image, haar)

        return boxes, image, backend

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > MAX_DIMENSION:
            scale = MAX_DIMENSION / float(longest)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def _detect_dnn(self, image: np.ndarray, net):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < DNN_CONFIDENCE_THRESHOLD:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes

    def _detect_haar(self, image: np.ndarray, haar):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = haar.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )
        return [tuple(map(int, f)) for f in faces]
