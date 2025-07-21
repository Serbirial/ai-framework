from flask import Flask, request, jsonify
from llama_cpp import Llama
from PIL import Image
import base64
import os
import tempfile
import cv2  # For extracting frames from video
import io
import requests
import easyocr

app = Flask(__name__)

# ---------------- OCR INITIALISATION ----------------
try:
    ocr_reader = easyocr.Reader(['en'], gpu=True)
except Exception as e:
    print(f"GPU OCR not available, falling back to CPU: {e}")
    ocr_reader = easyocr.Reader(['en'], gpu=False)

# ---------------- UTILS ----------------

def resize_image_pil(img: Image.Image, size=(512, 512)) -> Image.Image:
    """Resize keeping aspect ratio by letter‑boxing to exactly 512×512."""
    img.thumbnail(size, Image.LANCZOS)
    bg = Image.new("RGB", size, (0, 0, 0))
    bg.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return bg


def image_to_base64_data_uri(file_path: str) -> str:
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_data}"


def save_temp_resized_pil(pil_img: Image.Image) -> str:
    pil_img = resize_image_pil(pil_img)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    pil_img.save(tmp.name, quality=90)
    return tmp.name


def save_temp_resized_cv2(frame) -> str:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame)
    return save_temp_resized_pil(pil_img)

# ---------------- OBJECT DETECTION (OPTIONAL) ----------------

def get_objects_from_tflite_api(image_path):
    with open(image_path, "rb") as f:
        files = {"image": ("image.jpg", f, "image/jpeg")}
        try:
            response = requests.post("http://192.168.0.8:7007/detect_objects", files=files, timeout=5)
            response.raise_for_status()
            return response.json().get("objects", [])
        except Exception as e:
            print(f"Error calling TFLite detection API: {e}")
            return []

# ---------------- LLaMA MODEL ----------------
llm = Llama(
    model_path="SmolVLM-500M-Instruct-q8_0.gguf",
    n_ctx=800,
    n_threads=4,
    verbose=True,
)

# ---------------- ROUTES ----------------

@app.route("/describe_image", methods=["POST"])
def describe_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    image_path = save_temp_resized_pil(pil_img)

    detected_objects = None  # get_objects_from_tflite_api(image_path)  # optional
    ocr_text = " ".join([res[1] for res in ocr_reader.readtext(image_path)])
    image_data_uri = image_to_base64_data_uri(image_path)
    os.remove(image_path)

    prompt_text = "Describe this image."
    if detected_objects:
        objs_list = ", ".join(f"{o['label']} ({o['confidence']:.2f})" for o in detected_objects)
        prompt_text += f"\nDetected objects: {objs_list}."
    if ocr_text.strip():
        prompt_text += f"\nExtracted text: {ocr_text}"

    messages = [
        {
            "role": "system",
            "content": prompt_text
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}}
            ]
        }
    ]


    print(messages)
    try:
        resp = llm.create_chat_completion(messages=messages, max_tokens=600, temperature=0.5, repeat_penalty=1, top_p=0.9)
        content = resp["choices"][0]["message"]["content"]
    except Exception as e:
        content = f"Error: {e}"

    return jsonify({"result": content})


@app.route("/describe_video", methods=["POST"])
def describe_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    suffix = os.path.splitext(video_file.filename)[1]
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_path = tmp_video.name
    video_file.save(video_path)

    selected_frames, ocr_texts = [], []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every = int(fps * 1.5)
    idxs = [i * every for i in range(min(6, total // every))]

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        fpath = save_temp_resized_cv2(frame)
        selected_frames.append(fpath)
        ocr = ocr_reader.readtext(fpath)
        if ocr:
            ocr_texts.append(" ".join(r[1] for r in ocr))
    cap.release()
    os.remove(video_path)

    if not selected_frames:
        return jsonify({"error": "No suitable frames extracted"}), 400

    images_json = [{"type": "image_url", "image_url": {"url": image_to_base64_data_uri(p)}} for p in selected_frames]
    for p in selected_frames:
        os.remove(p)

    prompt = (
        "These frames are from a short video posted in a group chat. "
        "Summarize what happens, describe interesting moments or vibes, and react like you're chatting with friends." )
    if ocr_texts:
        prompt += "\nExtracted text: " + " ".join(ocr_texts)

    messages = [
        {"role": "system", "content": "You are in a casual chat reacting to videos."},
        {"role": "user", "content": images_json + [{"type": "text", "text": prompt}]}
    ]

    try:
        resp = llm.create_chat_completion(messages=messages, max_tokens=512)
        reply = resp["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"Error: {e}"

    return jsonify({"result": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 6006)))