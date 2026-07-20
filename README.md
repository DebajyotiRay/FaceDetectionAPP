# Attendance Register — Photo Roll Call

Upload one photo of a classroom. Get an instant headcount, with every
detected face marked, in under a second. No manual roll call.

This is a from-scratch rebuild of a BTech-era C++ / OpenCV face-detection
exercise (`FaceDetection.cpp`, 2021), turned into an actual usable web tool
for an M.Tech portfolio.

## What changed from the original project

| | BTech version (2021) | This version |
|---|---|---|
| Interface | Console app, hardcoded local file paths | Browser upload, drag-and-drop |
| Detector | Haar cascade only | DNN (res10 SSD) with automatic fallback to Haar if offline |
| Output | Saves an image file to disk | Live count + annotated photo in-browser |
| Scope | Detects faces in exactly 2 hardcoded images | Any single photo, any class size, resized automatically |
| Use case | Demo of `detectMultiScale` | Framed around a real problem: classroom attendance |

## How it works

1. You upload a photo (e.g. taken from the front of the classroom).
2. Flask receives the image, decodes it with OpenCV, and downsizes it if it's very large.
3. `face_detector.py` runs it through OpenCV's DNN face detector (a small
   Caffe-based SSD model) if the model files are available locally — this
   backend is noticeably better than Haar cascades at side-angled faces,
   uneven lighting, and smaller faces near the back of a room.
4. If the DNN model files can't be downloaded (e.g. no internet, first run
   offline), the app **automatically falls back** to a tuned Haar cascade —
   the same detector family as the original project, bundled with OpenCV,
   so the app still works with zero setup.
5. Detected faces are boxed and numbered on the photo, and the count is
   returned to the browser.

## Setup

```bash
cd attendance-counter
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

On first run, the app tries to download the DNN face detector model
(~10 MB, one-time, from OpenCV's own GitHub repo) into `models/`. If your
network blocks the download, it silently falls back to the Haar cascade —
no action needed either way. The active backend is shown on the page itself.

## Project structure

```
attendance-counter/
├── app.py              Flask routes: "/" (upload page) and "/detect" (API)
├── face_detector.py    Detection engine — DNN backend with Haar fallback
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── style.css        "Attendance register / ledger" visual theme
    └── script.js        Upload, drag-and-drop, results rendering
```

## Production-readiness notes

A few things were deliberately fixed to avoid rookie mistakes:

- **Debug mode is off by default.** Flask's debugger allows arbitrary code
  execution if left on and exposed — it's opt-in only, via `FLASK_DEBUG=1`.
- **Binds to `127.0.0.1` by default**, not `0.0.0.0`. Set `HOST=0.0.0.0` if
  you deliberately want other devices on the same network to reach it.
- **No stack traces are ever sent to the browser.** Errors are logged
  server-side and returned as clean JSON messages instead.
- **Oversized uploads (>12 MB) get a friendly error**, not Werkzeug's
  default HTML error page.
- **Model downloads have a timeout** and clean up partial files on failure,
  so a flaky network can't leave the app permanently stuck or hang startup.
- **`.gitignore`** keeps `__pycache__/`, `venv/`, and the downloaded model
  binaries out of version control.

## Known limitations (worth mentioning if you demo or discuss this)

- **Not a recognition system** — it counts and boxes faces, it does not
  identify who is present. That's a natural next step (see below).
- Very oblique side profiles, heavy backlighting, or masks reduce accuracy,
  as with any general-purpose face detector.
- The Haar fallback is faster but produces more false positives/negatives
  than the DNN backend — the app tells you which one is active so you can
  judge how much to trust the count.
- This is a development server (`app.run`), not a production deployment —
  fine for a demo or local classroom use, not for hosting publicly as-is.

## Ideas for extending this further

These are natural "v2" directions if you want to keep building on this for
your M.Tech coursework or a resume project:

- **Attendance matching**: pair detected faces against a class roster using
  face embeddings (e.g. `face_recognition` / ArcFace) to mark named students
  present/absent, not just a headcount.
- **CSV/Excel export**: log each session's count and timestamp for record-keeping.
- **Live webcam mode**: capture directly from a classroom camera instead of
  uploading a file.
- **Confidence/quality warnings**: flag photos that are too dark, blurry, or
  taken at too sharp an angle before relying on the count.
- **Accuracy benchmarking**: compare Haar vs. DNN precision/recall on a small
  labeled set of classroom photos — good material for a short report or
  even a mini research write-up.

## Tech stack

Python, Flask, OpenCV (DNN + Haar cascade), vanilla HTML/CSS/JS (no frontend
framework needed for a single-page tool like this).
