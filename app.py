from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import logging
from model import EfficientNet  # Adjust the import according to your file structure

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
model_version = 'b0'  # Change this according to your model version
model = EfficientNet(version=model_version)
model.load_state_dict(torch.load("model (2).pth", map_location=torch.device('cpu')))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        image = Image.open(file).convert('RGB')
    except Exception as e:
        logging.error(f"Error opening image: {e}")
        return jsonify({'error': 'Invalid image format'}), 400

    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    logging.debug(f"Transformed image shape: {image.shape}")

    with torch.no_grad():
        output = model(image)
        prediction = output.item()  # Directly get the scalar value if sigmoid is in the model
        logging.debug(f"Model output: {output}, Prediction: {prediction}")
        result = 'AI Image' if prediction > 0.5 else 'Real Image'
        confidence = prediction if result == 'AI Image' else 1 - prediction
        logging.debug(f"Prediction result: {result}, Confidence: {confidence}")

    return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
