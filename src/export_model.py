import torch
from src.model import MNISTModel

model = MNISTModel()
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location="cpu"))
model.eval()

example_input = torch.randn(1, 1, 28, 28)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("models/mnist_cnn_traced.pt")
print("✅ Modèle exporté : models/mnist_cnn_traced.pt")
