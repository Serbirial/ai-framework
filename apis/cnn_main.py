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

# Initialize OCR once, try GPU acceleration, fallback to CPU
try:
    ocr_reader = easyocr.Reader(['en'], gpu=True)
except Exception as e:
    print(f"GPU OCR not available, falling back to CPU: {e}")
    ocr_reader = easyocr.Reader(['en'], gpu=False)

def get_objects_from_tflite_api(image_path):
    with open(image_path, "rb") as f:
        files = {"image": ("image.jpg", f, "image/jpeg")}
        try:
            response = requests.post("http://192.168.0.8:7007/detect_objects", files=files, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("objects", [])
        except Exception as e:
            print(f"Error calling TFLite detection API: {e}")
            return []

# Initialize your LLaMA.cpp model
llm = Llama(
    model_path="SmolVLM-500M-Instruct-f16.gguf",
    n_ctx=1400,
    n_threads=4,
    verbose=True
)

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

@app.route("/describe_image", methods=["POST"])
def describe_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_path = tmp.name
            image = Image.open(file.stream).convert("RGB")
            image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400
    detected_objects = None
    try:
        #detected_objects = get_objects_from_tflite_api(image_path)
        ocr_text = " ".join([res[1] for res in ocr_reader.readtext(image_path)])
        image_data_uri = image_to_base64_data_uri(image_path)
    finally:
        os.remove(image_path)

    objects_descr = ""
    if detected_objects:
        objs_list = [f"{obj['label']} ({obj['confidence']:.2f})" for obj in detected_objects]
        objects_descr = "Detected objects: " + ", ".join(objs_list) + "."

    prompt_text = (
        "You are reacting to an image like a person in a chat room.\n"
        "Do not just describe it factually — describe any strange or unexpected behavior, relationships, jokes, power dynamics, or humor.\n"
        "Assume this image was posted online to be funny, shocking, or strange. Try to reason why.\n"
        "Use expressive language. If the image involves people in odd situations, react like a random chatter might.\n"
    )

    if objects_descr:
        prompt_text += f"\n{objects_descr}"

    if ocr_text.strip():
        prompt_text += f"\nExtracted text from the image: {ocr_text}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=600,
            temperature=0.5,
            repeat_penalty=1,
            top_p=0.9
        )
        content = response["choices"][0]["message"]["content"]
    except Exception as e:
        content = f"Error: {e}"

    return jsonify({"result": content})

@app.route("/describe_video", methods=["POST"])
def describe_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    filename = video_file.filename.lower()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            video_path = tmp.name
            video_file.save(video_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save video: {e}"}), 400

    selected_frames = []
    ocr_texts = []

    try:
        if filename.endswith(".gif"):
            gif = Image.open(video_path)
            frames = []
            try:
                while True:
                    frames.append(gif.copy().convert("RGB"))
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass

            step = max(1, len(frames) // 6)
            frames_to_use = frames[::step][:6]

            for frame in frames_to_use:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_frame:
                    frame.save(tmp_frame.name)
                    selected_frames.append(tmp_frame.name)

        elif filename.endswith(".mp4") or filename.endswith(".webm"):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 10
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(fps * 1.5)
            frame_indices = [int(i * frame_interval) for i in range(min(6, int(total_frames // frame_interval)))]

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as frame_file:
                    cv2.imwrite(frame_file.name, frame)
                    selected_frames.append(frame_file.name)
            cap.release()
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        os.remove(video_path)

        if not selected_frames:
            return jsonify({"error": "No usable frames found"}), 400

        images_json = []
        for path in selected_frames:
            uri = image_to_base64_data_uri(path)
            images_json.append({"type": "image_url", "image_url": {"url": uri}})
            ocr_result = ocr_reader.readtext(path)
            if ocr_result:
                ocr_texts.append(" ".join([res[1] for res in ocr_result]))
            os.remove(path)

        combined_ocr = "\n".join(ocr_texts).strip()

    except Exception as e:
        return jsonify({"error": f"Failed to process video: {e}"}), 500

    prompt_text = (
        "These frames are taken from a short video posted in a group chat.\n"
        "Summarize what happens, describe any interesting moments or vibes, "
        "and react naturally like you're chatting with friends. You can joke, emote, or analyze if it's a meme or event."
    )

    if combined_ocr:
        prompt_text += f"\n\nExtracted text from the video frames:\n{combined_ocr}"

    messages = [
        {"role": "system", "content": "You are in a casual chat reacting to videos. Be expressive and human."},
        {
            "role": "user",
            "content": images_json + [{"type": "text", "text": prompt_text}]
        }
    ]

    try:
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=512
        )
        reply = result["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"Error: {e}"

    return jsonify({"result": reply})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6006))
    app.run(host="0.0.0.0", port=port)
