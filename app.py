from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS  # ✅ NEW
import os
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load("../ai_model/model.pth", map_location=torch.device('cpu'), weights_only=False)
model.eval()

# Update class names to match your dataset
classes = ['eczema', 'ringworm', 'psoriasis']

# Flask app setup
app = Flask(__name__)
CORS(app)  # ✅ ENABLE cross-origin requests (needed when serving frontend via http.server)
UPLOAD_FOLDER = "../uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    input_tensor = preprocess_image(filepath)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]

    return jsonify({"prediction": label})

if __name__ == '__main__':
    app.run(debug=True)
