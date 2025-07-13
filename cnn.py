from flask import Flask, request, jsonify
from llama_cpp import Llama
from PIL import Image
import base64
import os
import tempfile
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
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_path = tmp.name
            image = Image.open(file.stream).convert("RGB")
            image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    # Convert image to base64 data URI
    try:
        image_data_uri = image_to_base64_data_uri(image_path)
    finally:
        os.remove(image_path)  # Clean up temp file

    # Prepare the prompt using multimodal message format
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

    # Generate chat completion
    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512
        )
        content = response["choices"][0]["message"]["content"]
    except Exception as e:
        content = f"Error: {e}"

    return jsonify({"result": content})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6006))
    app.run(host="0.0.0.0", port=port)
