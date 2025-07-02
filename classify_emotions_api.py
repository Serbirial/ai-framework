from flask import Flask, request, jsonify
from llama_cpp import Llama
import src.classify as classify
from src.static import emotionalLLMName, baseclassifyLLMName

base_model = Llama(
    model_path=baseclassifyLLMName,
    n_ctx=512,
    n_threads=4,
    logits_all=False,
    use_mmap=False,
    n_gpu_layers=0,
    low_vram=True,
    n_batch=4,
    numa=False
    )

app = Flask(__name__)


@app.route('/classify_likes_dislikes', methods=['POST'])
def classify_likes_dislikes():
    data = request.get_json()
    user_input = data.get('user_input', '')
    likes = data.get('likes', [])
    dislikes = data.get('dislikes', [])

    classification = classify.classify_likes_dislikes_user_input(base_model, None, user_input, likes, dislikes)
    return jsonify({"classification": classification})


@app.route('/classify_social_tone', methods=['POST'])
def classify_social_tone():
    data = request.get_json()
    user_input = data.get('user_input', '')

    classification = classify.classify_social_tone(base_model, None, user_input)
    return jsonify({"classification": classification})


@app.route('/determine_moods', methods=['POST'])
def determine_moods():
    data = request.get_json()
    classification = data.get('classification', {})
    top_n = data.get('top_n', 3)

    moods = classify.determine_moods_from_social_classification(base_model, top_n=top_n)
    return jsonify({"top_moods": moods})


@app.route('/classify_moods_into_sentence', methods=['POST'])
def classify_moods_into_sentence():
    data = request.get_json()
    moods_dict = data.get('moods_dict', {})

    mood_sentence = classify.classify_moods_into_sentence(base_model, None, moods_dict)
    return jsonify({"mood_sentence": mood_sentence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, threaded=False)
