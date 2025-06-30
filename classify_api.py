

web_prompt = "Summarize the following content in plain language, including key facts, values, and structure:"

from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch
import src.classify as classify  # import the whole classify.py file as a module
from src.static import classifyLLMName


# Load TinyT5 model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(classifyLLMName)
model = ORTModelForSeq2SeqLM.from_pretrained(classifyLLMName, export=False)


app = Flask(__name__)

@app.route('/classify_user_input', methods=['POST'])
def classify_user_input():
    data = request.get_json()
    user_input = data.get('user_input', '')

    category = classify.classify_user_input(model, tokenizer, user_input)
    return jsonify({"category": category})


@app.route('/classify_likes_dislikes', methods=['POST'])
def classify_likes_dislikes():
    data = request.get_json()
    user_input = data.get('user_input', '')
    likes = data.get('likes', [])
    dislikes = data.get('dislikes', [])

    classification = classify.classify_likes_dislikes_user_input(model, None, user_input, likes, dislikes)
    return jsonify({"classification": classification})


@app.route('/classify_social_tone', methods=['POST'])
def classify_social_tone():
    data = request.get_json()
    user_input = data.get('user_input', '')

    classification = classify.classify_social_tone(model, tokenizer, user_input)
    return jsonify({"classification": classification})


@app.route('/determine_moods_social', methods=['POST'])
def determine_moods():
    data = request.get_json()
    classification = data.get('classification', {})
    top_n = data.get('top_n', 3)

    moods = classify.determine_moods_from_social_classification(classification, top_n=top_n)
    return jsonify({"top_moods": moods})


@app.route('/classify_moods_into_sentence', methods=['POST'])
def classify_moods_into_sentence():
    data = request.get_json()
    moods_dict = data.get('moods_dict', {})
    model = classify.model
    tokenizer = classify.tokenizer

    mood_sentence = classify.classify_moods_into_sentence(model, tokenizer, moods_dict)
    return jsonify({"mood_sentence": mood_sentence})


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

    summary = classify.summarize_raw_scraped_data(model, input_text, max_tokens=max_tokens)
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
