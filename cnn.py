from flask import Flask, request, jsonify
from llama_cpp import Llama
import os

app = Flask(__name__)
# wget https://huggingface.co/Mungert/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-q5_0_l.gguf
# Load the VLM once at startup
llm = Llama(
    model_path="smolvlm-500m-instruct.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    logits_all=False,
    verbose=False
)

@app.route("/describe_image", methods=["POST"])
def describe_image():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    image_file = request.files["image"]
    prompt_text = request.form.get("prompt", "Describe this image as if you're chatting with a friend online.")

    # Save image temporarily
    temp_path = image_file.filename
    image_file.save(temp_path)

    # Compose prompt
    messages = [
        {"role": "system", "content": "You are a friendly assistant in a chat room who casually describes images."},
        {"role": "user", "content": f"<image>\n{prompt_text.strip()}"}
    ]

    # Run inference
    result = llm.create_chat_completion(
        messages=messages,
        image_path=temp_path
    )

    # Remove temp image
    os.remove(temp_path)

    return jsonify({
        "description": result["choices"][0]["message"]["content"].strip()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6006)
