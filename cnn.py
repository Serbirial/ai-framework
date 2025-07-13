from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from PIL import Image
import torch
import os

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor once
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager"
).to(DEVICE)

@app.route("/describe_image", methods=["POST"])
def describe_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "You're in a lively, informal group chat where someone just dropped this image with no explanation. "
                        "It might be a photo, selfie, meme, or something random. React like a real person would—funny, sarcastic, curious, flirty, whatever fits. "
                        "If it's a meme, try to understand the joke and respond naturally, as if you're talking to close friends. "
                        "Be descriptive, expressive, and human. Mention what stands out, what the vibe is, and how you'd reply in the chat."
                    )
                }
            ]
        }
    ]


    # Prepare prompt
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(result)
    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6006))
    app.run(host="0.0.0.0", port=port)
