from flask import Flask, request, jsonify
from llama_cpp import Llama
import src.classify as classify
from src.static import baseclassifyLLMName

model = Llama(
    model_path=baseclassifyLLMName,
    n_ctx=512,
    n_threads=3,  # tune per device
    verbose=False
)

web_prompt = "Summarize the following content in plain language, including key facts, values, and structure: "

app = Flask(__name__)


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
