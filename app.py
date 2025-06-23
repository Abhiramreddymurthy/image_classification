from flask import Flask, request, jsonify
from inference import predict_image
import os

app = Flask(__name__)
class_names = sorted(os.listdir("dataset"))  # Assuming folder names are class labels

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image = request.files['file']
    image_path = os.path.join("temp.jpg")
    image.save(image_path)

    result = predict_image(image_path, class_names=class_names)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
