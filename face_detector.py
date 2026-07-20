"""
face_detector.py
-----------------
Face detection engine for the classroom attendance counter.

Two backends are supported:
  1. DNN  - OpenCV's res10 SSD Caffe face detector. Much more accurate,
            handles angled faces, varied lighting, partial occlusion, and
            small/far-away faces (all common in a classroom photo).
  2. Haar - Classic Viola-Jones cascade (the same family of detector used
            in the original 2021 BTech project). Zero download required,
            since it ships inside the opencv-python package.

On startup we try to fetch the DNN model files into ./models the first
time the app runs (they're ~10 MB, one-time download from OpenCV's own
GitHub repo). If that fails for any reason (no internet, blocked network,
etc.) we transparently fall back to Haar so the app still works offline.
"""

import os
import socket
import urllib.request
import cv2
import numpy as np

DOWNLOAD_TIMEOUT_SECONDS = 6

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
        self.backend = None
        self.net = None
        self.haar = None
        self._load_backend()

    # ------------------------------------------------------------------ #
    # Backend setup
    # ------------------------------------------------------------------ #
    def _load_backend(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        if self._ensure_dnn_files():
            try:
                self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
                self.backend = "dnn"
                return
            except Exception as e:
                print(f"[face_detector] Failed to load DNN model, falling back to Haar: {e}")

        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.haar = cv2.CascadeClassifier(cascade_path)
        if self.haar.empty():
            raise RuntimeError("Could not load Haar cascade classifier — OpenCV install looks broken.")
        self.backend = "haar"

    def _ensure_dnn_files(self) -> bool:
        """Return True if both DNN model files are present locally (downloading if needed)."""
        try:
            self._download_if_missing(PROTOTXT_URL, PROTOTXT_PATH)
            self._download_if_missing(CAFFEMODEL_URL, CAFFEMODEL_PATH)
            return os.path.exists(PROTOTXT_PATH) and os.path.exists(CAFFEMODEL_PATH)
        except Exception as e:
            print(f"[face_detector] Could not download DNN model files ({e}). Using Haar cascade instead.")
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
        Returns (list_of_boxes, backend_name) where each box is (x, y, w, h).
        """
        image = self._resize_if_needed(image)

        if self.backend == "dnn":
            boxes = self._detect_dnn(image)
        else:
            boxes = self._detect_haar(image)

        return boxes, image, self.backend

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > MAX_DIMENSION:
            scale = MAX_DIMENSION / float(longest)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def _detect_dnn(self, image: np.ndarray):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

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

    def _detect_haar(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )
        return [tuple(map(int, f)) for f in faces]
