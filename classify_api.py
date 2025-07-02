

web_prompt = "Summarize the following content in plain language, including key facts, values, and structure:"

from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch
import src.classify as classify  # import the whole classify.py file as a module
from src.static import classifyLLMName


# Load TinyT5 model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(classifyLLMName, use_fast=True)
model = ORTModelForSeq2SeqLM.from_pretrained(classifyLLMName, export=False)


app = Flask(__name__)

@app.route('/classify_user_input', methods=['POST'])
def classify_user_input():
    data = request.get_json()
    user_input = data.get('user_input', '')

    category = classify.classify_user_input(model, tokenizer, user_input)
    return jsonify({"category": category})



@app.route('/extract_search_query', methods=['POST'])
def extract_search_query():
    data = request.get_json()
    input_text = data.get('input_text', '')
    role = data.get('role', 'user')
    model = classify.model

    query = classify.extract_search_query_llama(model, input_text, role=role)
    return jsonify({"search_query": query})


@app.route('/summarize_detailed', methods=['POST'])
def classify_summarize_input():
    data = request.get_json()
    input_text = data.get('text', '')
    max_tokens = data.get('max_tokens', 200)
    model = classify.model

    summary = classify.classify_summarize_input(model, input_text, max_tokens=max_tokens)
    return jsonify({"summary": summary})

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    input_text = data.get("text", "")
    
    if not input_text:
        return jsonify({"error": "Missing 'text' field"}), 400

    inputs = tokenizer(web_prompt + input_text, return_tensors="pt", max_length=300, truncation=True)
    
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
