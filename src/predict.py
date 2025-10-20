import torch
from src.model import MNISTModel
import torch.nn.functional as F

# Charger le modèle une seule fois au lancement du module
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/mnist_cnn.pth"

model = MNISTModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict(tensor_img):
    with torch.no_grad():
        tensor_img = tensor_img.to(device)
        output = model(tensor_img)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
        all_probs = probs.cpu().numpy().flatten().tolist()
    return predicted_class, confidence, all_probs