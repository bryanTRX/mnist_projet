from flask import Blueprint, request, jsonify
from PIL import Image
from app.utils import preprocess_canvas_image
from app.predict import predict
import numpy as np

main = Blueprint('main', __name__)

@main.route("/status", methods=["GET"])
def status():
    return jsonify({"status":"ok"})

@main.route("/preview", methods=["POST"])
def preview():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    canvas_data = np.array(Image.open(file).convert("L"))
    tensor_img = preprocess_canvas_image(canvas_data)
    return jsonify({"shape": list(tensor_img.shape)})

@main.route("/predict", methods=["POST"])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    canvas_data = np.array(Image.open(file).convert("L"))
    tensor_img = preprocess_canvas_image(canvas_data)
    predicted_class, confidence, all_probs = predict(tensor_img)
    
    return jsonify({
        "prediction": predicted_class, 
        "confidence": confidence,
        "probabilities": all_probs            
        })