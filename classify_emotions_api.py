

web_prompt = "Summarize the following content in plain language, including key facts, values, and structure:"

from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch
import src.classify as classify  # import the whole classify.py file as a module
from src.static import emotionalLLMName


# Load TinyT5 model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(emotionalLLMName, use_fast=True)
model = ORTModelForSeq2SeqLM.from_pretrained(emotionalLLMName, export=False)


app = Flask(__name__)


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


@app.route('/determine_moods', methods=['POST'])
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



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, threaded=False)
