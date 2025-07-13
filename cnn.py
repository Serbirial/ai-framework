from flask import Flask, request, jsonify
from llama_cpp import Llama
from PIL import Image
import base64
import os
import tempfile
import cv2  # For extracting frames from video
import io

app = Flask(__name__)

# Initialize your LLaMA.cpp model
llm = Llama(
    model_path="SmolVLM2-500M-Video-Instruct-Q3_K_S.gguf",
    n_ctx=1024,
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

    try:
        image_data_uri = image_to_base64_data_uri(image_path)
    finally:
        os.remove(image_path)

    messages = [
        {"role": "system", "content": "You are a casual assistant in a group chat. Be expressive, funny, sarcastic, or flirty if it fits."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {
                    "type": "text",
                    "text": (
                        "You're in a lively, informal group chat where someone just dropped this image with no explanation. "
                        "React like a real person would—funny, sarcastic, curious, flirty, whatever fits."
                    )
                }
            ]
        }
    ]

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512
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
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_path = tmp.name
            video_file.save(video_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save video: {e}"}), 400

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        frame_interval = int(fps * 1.5)  # Grab 1 frame every ~1.5 seconds

        max_frames = 6  # limit for performance
        selected_frames = []
        frame_indices = [int(i * frame_interval) for i in range(min(max_frames, int(duration_sec // 1.5)))]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as frame_file:
                frame_path = frame_file.name
                cv2.imwrite(frame_path, frame)
                selected_frames.append(frame_path)

        cap.release()
        os.remove(video_path)

        if not selected_frames:
            return jsonify({"error": "No usable frames found"}), 400

        # Convert all frames to base64
        images_json = []
        for path in selected_frames:
            uri = image_to_base64_data_uri(path)
            images_json.append({"type": "image_url", "image_url": {"url": uri}})
            os.remove(path)

    except Exception as e:
        return jsonify({"error": f"Failed to process video: {e}"}), 500

    # Build full prompt with all sampled frames
    messages = [
        {"role": "system", "content": "You are in a casual chat reacting to videos. Be expressive and human."},
        {
            "role": "user",
            "content": images_json + [
                {
                    "type": "text",
                    "text": (
                        "These frames are taken from a short video posted in a group chat. "
                        "Summarize what happens in the video, describe any interesting moments or vibes, "
                        "and react naturally like you're chatting with friends. You can joke, emote, or analyze if it's a meme or event."
                    )
                }
            ]
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
