from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import base64
import tempfile
import os
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# COCO labels for SSD MobileNet v1
LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Load TFLite model
interpreter = tflite.Interpreter(model_path="~/models/cnn/detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((300, 300))
    input_data = np.expand_dims(np.array(image, dtype=np.uint8), axis=0)
    return input_data

@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_path = tmp.name
            image = Image.open(file.stream).convert("RGB")
            image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    try:
        input_data = preprocess_image(image_path)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        count = int(interpreter.get_tensor(output_details[3]['index'])[0])

        results = []
        for i in range(count):
            if scores[i] < 0.4:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            results.append({
                "label": LABELS[int(class_ids[i])],
                "confidence": float(scores[i]),
                "bbox": {
                    "xmin": float(xmin),
                    "ymin": float(ymin),
                    "xmax": float(xmax),
                    "ymax": float(ymax)
                }
            })
    finally:
        os.remove(image_path)

    return jsonify({"objects": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7007))
    app.run(host="0.0.0.0", port=port)
