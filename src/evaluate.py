import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import MNISTModel

def evaluate(model_path: str = "models/mnist_cnn.pth", data_path: str = "data/test", batch_size: int = 64, device: str = "cpu"):
    model = MNISTModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    test_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluation en cours", ncols=100):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total 
    print(f"\nRésultats du test :")
    print(f"    Test Loss     : {avg_loss:.4f}")
    print(f"    Test Accuracy : {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité terrain")
    plt.title("Matrice de confusion - MNIST CNN")
    plt.show(block=False)
    plt.savefig("results.png")
    plt.close()

    print("\nRapport de classification :")
    print(classification_report(all_labels, all_preds, digits=4))

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(device=device)
