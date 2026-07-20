const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const dropzoneInner = document.getElementById("dropzoneInner");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submitBtn");
const errorMsg = document.getElementById("errorMsg");

const resultRow = document.getElementById("resultRow");
const resultImage = document.getElementById("resultImage");
const odometer = document.getElementById("odometer");
const tallyCaption = document.getElementById("tallyCaption");
const resetBtn = document.getElementById("resetBtn");

let selectedFile = null;

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = !message;
}

function setSelectedFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("That doesn't look like an image file.");
    return;
  }
  selectedFile = file;
  showError("");

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropzoneInner.hidden = true;
  };
  reader.readAsDataURL(file);

  submitBtn.disabled = false;
}

fileInput.addEventListener("change", (e) => setSelectedFile(e.target.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  setSelectedFile(file);
});

submitBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  submitBtn.disabled = true;
  submitBtn.classList.add("loading");
  submitBtn.textContent = "Counting…";
  showError("");

  const formData = new FormData();
  formData.append("photo", selectedFile);

  try {
    const res = await fetch("/detect", { method: "POST", body: formData });

    let data;
    try {
      data = await res.json();
    } catch {
      showError("Unexpected response from the server. Please try again.");
      return;
    }

    if (!res.ok) {
      showError(data.error || "Something went wrong. Please try another photo.");
      return;
    }

    renderResult(data);
  } catch (err) {
    showError("Could not reach the server. Is it still running?");
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove("loading");
    submitBtn.textContent = "Take Attendance";
  }
});

function renderResult(data) {
  resultImage.src = data.image;
  resultRow.hidden = false;
  resultRow.scrollIntoView({ behavior: "smooth", block: "nearest" });

  renderOdometer(data.count);

  const noun = data.count === 1 ? "student" : "students";
  const engine = data.backend === "dnn" ? "high-accuracy detector" : "offline detector";
  tallyCaption.textContent = `${data.count} ${noun} detected in ${data.elapsed_ms} ms, using the ${engine}. Numbers on the photo match the order faces were found.`;
}

function renderOdometer(count) {
  odometer.innerHTML = "";
  const digits = String(count).split("");
  digits.forEach((d, i) => {
    const span = document.createElement("span");
    span.className = "digit";
    span.textContent = d;
    span.style.animationDelay = `${i * 60}ms`;
    odometer.appendChild(span);
  });
}

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  preview.hidden = true;
  dropzoneInner.hidden = false;
  submitBtn.disabled = true;
  resultRow.hidden = true;
  showError("");
  window.scrollTo({ top: 0, behavior: "smooth" });
});
