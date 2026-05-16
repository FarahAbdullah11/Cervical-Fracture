import torch
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

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
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# RANDOM IMAGE SELECTION
# =========================

fracture_folder = "../../cervical fracture/val/fracture"
normal_folder = "../../cervical fracture/val/normal"

all_images = []

for folder in [fracture_folder, normal_folder]:

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)

        all_images.append(img_path)

# Select 4 random images
selected_images = random.sample(all_images, 4)

# =========================
# CREATE FIGURE
# =========================

fig, ax = plt.subplots(4, 2, figsize=(10, 16))

# =========================
# PROCESS EACH IMAGE
# =========================

for i, image_path in enumerate(selected_images):

    print("Selected Image:", image_path)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Enable gradients
    input_tensor.requires_grad_()

    # =========================
    # FORWARD PASS
    # =========================

    output = model(input_tensor)

    predicted_class = output.argmax(dim=1)

    score = output[0, predicted_class]

    # =========================
    # BACKWARD PASS
    # =========================

    model.zero_grad()

    score.backward()

    # =========================
    # SALIENCY MAP
    # =========================

    saliency = input_tensor.grad.data.abs()

    saliency, _ = torch.max(saliency, dim=1)

    saliency = saliency.squeeze().cpu().numpy()

    # =========================
    # PLOT ORIGINAL IMAGE
    # =========================

    ax[i, 0].imshow(image)

    ax[i, 0].set_title("Original X-ray")

    ax[i, 0].axis("off")

    # =========================
    # PLOT SALIENCY MAP
    # =========================

    ax[i, 1].imshow(saliency, cmap='hot')

    ax[i, 1].set_title("Saliency Map")

    ax[i, 1].axis("off")

# =========================
# SHOW RESULTS
# =========================

plt.tight_layout()

plt.show(block=True)