from flask import Flask, request, jsonify
from llama_cpp import Llama
import src.classify as classify
from src.static import baseclassifyLLMName

model = Llama(
    model_path=baseclassifyLLMName,
    n_ctx=1024,
    n_threads=4,  # tune per device
    verbose=False,
    logits_all=False,
    use_mmap=False,
    n_gpu_layers=0,
    low_vram=True,
    n_batch=4,
    numa=False
)

web_prompt = "Summarize the following content in plain language, including key facts, values, and structure: "

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

    classification = classify.classify_social_tone(model, None, user_input)
    return jsonify({"classification": classification})


@app.route('/determine_moods', methods=['POST'])
def determine_moods():
    data = request.get_json()
    classification = data.get('classification', {})
    top_n = data.get('top_n', 3)

    moods = classify.determine_moods_from_social_classification(model, top_n=top_n)
    return jsonify({"top_moods": moods})


@app.route('/classify_moods_into_sentence', methods=['POST'])
def classify_moods_into_sentence():
    data = request.get_json()
    moods_dict = data.get('moods_dict', {})

    mood_sentence = classify.classify_moods_into_sentence(model, moods_dict)
    return jsonify({"mood_sentence": mood_sentence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, threaded=False)


@app.route('/classify_user_input', methods=['POST'])
def classify_user_input():
    data = request.get_json()
    user_input = data.get('user_input', '')
    category = classify.classify_user_input(model, None, user_input)
    return jsonify({"category": category})


@app.route('/extract_search_query', methods=['POST'])
def extract_search_query():
    data = request.get_json()
    input_text = data.get('input_text', '')
    role = data.get('role', 'user')
    query = classify.extract_search_query_llama(model, input_text, role=role)
    return jsonify({"search_query": query})


@app.route('/summarize_detailed', methods=['POST'])
def classify_summarize_input():
    data = request.get_json()
    input_text = data.get('text', '')
    max_tokens = data.get('max_tokens', 200)
    summary = classify.classify_summarize_input(model, input_text, max_tokens=max_tokens)
    return jsonify({"summary": summary})


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    input_text = data.get("text", "")
    if not input_text:
        return jsonify({"error": "Missing 'text' field"}), 400

    full_prompt = web_prompt + input_text
    response = model(full_prompt, max_tokens=128, stop=["\n"])
    summary = response["choices"][0]["text"].strip()
    return jsonify({"summary": summary})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, threaded=False)
