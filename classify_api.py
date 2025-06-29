from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch

app = Flask(__name__)

web_prompt = "Summarize the following content in plain language, including key facts, values, and structure:"

# Load TinyT5 model + tokenizer
model_name = "mrm8488/t5-small-finetuned-summarize-news"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ORTModelForSeq2SeqLM.from_pretrained(model_name, export=False)

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    input_text = data.get("text", "")
    
    if not input_text:
        return jsonify({"error": "Missing 'text' field"}), 400

    inputs = tokenizer(web_prompt + input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, threaded=False)
