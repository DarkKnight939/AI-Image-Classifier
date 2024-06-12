import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import EfficientNet  # Adjust the import according to your file structure
import requests
from io import BytesIO

# Download the model
model_url = 'https://github.com/DarkKnight939/AI-Image-Classifier/releases/download/models/model.2.pth'
response = requests.get(model_url)
model_data = BytesIO(response.content)

# Load the model
model_version = 'b0'  # Change this according to your model version
model = EfficientNet(version=model_version)
model.load_state_dict(torch.load(model_data, map_location=torch.device('cpu')))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit app
st.title("AI Image Classifier")
st.write("Upload an image to classify if it's an AI-generated image or a real image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.item()  # Directly get the scalar value if sigmoid is in the model
        result = 'AI Image' if prediction > 0.5 else 'Real Image'
        confidence = prediction if result == 'AI Image' else 1 - prediction

    # Display results
    st.write(f"**Result**: {result}")
    st.write(f"**Confidence**: {confidence * 100:.2f}%")
