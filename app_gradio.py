import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
LABELS_FR = {
    "cardboard": "📦 Carton",
    "glass":     "🍾 Verre",
    "metal":     "🥫 Métal",
    "paper":     "📄 Papier",
    "plastic":   "🧴 Plastique",
    "trash":     "🗑️ Ordures"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 6)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

def predict(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return {LABELS_FR[CLASSES[i]]: round(probs[i].item(), 3) for i in range(6)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="♻️ Tri des déchets",
    description="Upload une photo d'un déchet → le modèle identifie la bonne poubelle."
)

demo.launch()