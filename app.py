from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

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

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><body style="font-family:sans-serif;max-width:500px;margin:40px auto">
    <h2>♻️ Tri des déchets</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*"><br><br>
        <button type="submit">Analyser</button>
    </form>
    </body></html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    label = CLASSES[pred.item()]
    confidence = round(conf.item() * 100, 1)
    return f"""
    <html><body style="font-family:sans-serif;max-width:500px;margin:40px auto">
    <h2>♻️ Résultat</h2>
    <p>Classe : <b>{label}</b></p>
    <p>Confiance : <b>{confidence}%</b></p>
    <a href="/">← Réessayer</a>
    </body></html>
    """