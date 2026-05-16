import torch
from torchvision import transforms
from PIL import Image
import timm

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================

model = timm.create_model(
    'vit_tiny_patch16_224',
    pretrained=False,
    num_classes=2
)

# Load trained weights
model.load_state_dict(torch.load("../../vit_model.pth"))

model = model.to(device)

model.eval()

# =========================
# IMAGE PREPROCESSING
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# LOAD IMAGE
# =========================

image_path = r"F:\SPRG-26\ANN\ANN Project\Cervical-Fracture\cervical fracture\train\normal\CSFDV1B10 (110)-rotated1-rotated3.png"

image = Image.open(image_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

# =========================
# PREDICTION
# =========================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

# =========================
# CLASS NAMES
# =========================

classes = ['fracture', 'normal']

predicted_class = classes[predicted.item()]

confidence_score = confidence.item() * 100

# =========================
# RESULTS
# =========================

print(f"\nPrediction: {predicted_class}")

print(f"Confidence: {confidence_score:.2f}%")