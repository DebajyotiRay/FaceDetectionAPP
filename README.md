# Attendance Register — Photo Roll Call

Take one photo of the classroom, upload it, and get a headcount back in
under a second — with every face marked so you can double check it.

Manual roll call in a lecture hall of 60+ students eats real time every
single class. This is meant to replace that with a photo.

## How it works

You upload a photo, Flask picks it up and hands it to OpenCV, and OpenCV
finds the faces. Under the hood there are actually two possible detectors:

- By default it tries to use OpenCV's DNN face detector (a small SSD model),
  which is noticeably better than older methods at catching angled faces,
  bad lighting, and small faces near the back of the room.
- If that model can't be downloaded for some reason — no internet, a
  blocked network, whatever — it just falls back to a Haar cascade instead.
  That one ships with OpenCV already, so the app never breaks, it just gets
  a bit less accurate. The page tells you which one is currently active.

Once faces are found, they get boxed and numbered on the photo, and the
count gets sent back to the browser.

## Running it

```bash
cd attendance-counter
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 and you're in.

First time you run it, it'll try to grab the DNN model (~10MB) from
OpenCV's GitHub. If that fails it just quietly switches to the Haar
cascade — nothing you need to do either way.

## Project structure

```
attendance-counter/
├── app.py              Flask routes — the upload page and the /detect API
├── face_detector.py    Detection logic, DNN with a Haar fallback
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── style.css        the "ledger" look
    └── script.js        upload handling, drag-and-drop, results
```

## A few deliberate choices

Debug mode is off by default — Flask's debugger lets you run arbitrary
code if it's left on and exposed, so it's opt-in only (`FLASK_DEBUG=1`).
Same idea with the host: it binds to `127.0.0.1` unless you explicitly set
`HOST=0.0.0.0`.

Errors don't leak stack traces to the browser — they get logged on the
server and the user just sees a plain message. Uploads over 12MB get a
proper error instead of Flask's default ugly page. And the model download
has a timeout, so a bad network fails fast into the Haar fallback instead
of hanging the app on startup.

## Where it struggles

It's a face detector, not a face recognizer — it'll tell you how many
people are in the photo, not who they are. Extreme side angles, heavy
backlighting, or anything covering half a face (masks, hands, hair) can
throw it off, same as any general-purpose detector would struggle. The
Haar fallback in particular is faster but noisier than the DNN model, which
is why the app always shows you which one it's using — worth a glance
before trusting the number blindly.

This also runs on Flask's built-in dev server, which is fine for a demo or
local use but isn't meant to be exposed publicly as-is.

## Built with

Python, Flask, OpenCV, and plain HTML/CSS/JS — no frontend framework
needed for something this size.
