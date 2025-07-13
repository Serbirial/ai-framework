from flask import Flask, request, jsonify
from llama_cpp import Llama
from PIL import Image
import os
import tempfile
from utils.openai import extract_generated_text

app = Flask(__name__)

# Load vision-capable LLaMA.cpp model
llm = Llama(
    model_path="smolvlm-500m-instruct.Q4_K_M.gguf",
    n_ctx=1024,
    n_threads=4,
    verbose=True
)

@app.route("/describe_image", methods=["POST"])
def describe_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_path = tmp.name
            image = Image.open(file.stream).convert("RGB")
            image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    # Prepare the prompt
    messages = [
        {
            "role": "system",
            "content": "You are a casual assistant in a group chat. Be expressive, funny, sarcastic, or flirty if it fits."
        },
        {
            "role": "user",
            "content": "<image>\nYou're in a lively, informal group chat where someone just dropped this image with no explanation. React like a real person would—funny, sarcastic, curious, flirty, whatever fits."
        }
    ]

    # Generate with vision input
    try:
        result = llm.create_chat_completion(
            messages=messages,
            image_path=image_path,
            max_tokens=512
        )
        reply = result["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"Error during generation: {e}"

    # Clean up temp file
    os.remove(image_path)

    return jsonify({"result": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6006))
    app.run(host="0.0.0.0", port=port)
