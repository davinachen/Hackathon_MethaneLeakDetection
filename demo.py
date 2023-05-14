from utils import *
import streamlit as st
from streamlit_folium import st_folium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image

# Load previously trained model 
model = torch.load('./data/models/ResNet_pretrained_resnet50_lr1em3.pt')

# Test image transformation
image_transforms = { 
    'test': transforms.Compose([
        transforms.CenterCrop(size=64),
        transforms.ToTensor()
    ])
}

# Define prediction function
def predict(input, model):
    # Evaluation mode
    model.eval()
    # Add batch dimension 
    input = input.unsqueeze(0)
    # Get predicted category for image
    with torch.no_grad():
        outputs = model(input)
        ret, prediction = torch.max(outputs.data, 1)
    return prediction

# Streamlit Interface Building
st.title('Methane Detection')
st.header('Upload an image for methane leaks detection:')
input = st.file_uploader("Choose a file", type='tif')
if input is not None:
    # Get the relative path of the file
    file_name = input.name
    path = os.path.join('./data/test_images/', file_name)
    image = Image.open(path).convert('RGB')
    image = image_transforms['test'](image)
    # Make prediction
    prediction = predict(image, model)
    result = 'Yes' if prediction == 1 else 'No'
    st.write("prediction of methane leak: "+ result)