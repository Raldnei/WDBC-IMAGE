import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn.functional as F

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  
model.load_state_dict(torch.load('modelo_treinadoRESNET50.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), probabilities[0][0].item() * 100, probabilities[0][1].item() * 100  

st.title("Classificador de Câncer de Mama")
st.write("Envie uma imagem para determinar se é benigno ou maligno.")

uploaded_image = st.file_uploader("Escolha uma imagem...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagem enviada", use_column_width=True)
    
    prediction, benign_confidence, malignant_confidence = predict(image)
    
    min_confidence = 51  
    
    if prediction == 0:
        if benign_confidence >= min_confidence:
            st.write(f"O tumor é benigno com {benign_confidence:.2f}% de certeza.")
           
        else:
            st.write(f"A IA não tem certeza suficiente para afirmar que o tumor é benigno.Pois a Confiança é de apenas: {benign_confidence:.2f}%")
           
    else:
        if malignant_confidence >= min_confidence:
            st.write(f"O tumor é maligno com {malignant_confidence:.2f}% de certeza.")
        else:   
            st.write(f"A IA não tem certeza suficiente para afirmar que o tumor é maligno. Pois a Confiança é de apenas: {malignant_confidence:.2f}%")
            
