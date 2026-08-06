import torch
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import load_metadata, create_dataloaders
from model import create_model

sns.set_style("darkgrid")

def evaluate_model():
    """Evaluate the trained ResNet-18 model."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    
    metadata = load_metadata()
    _, test_loader = create_dataloaders(metadata)
    
    model = create_model()

    model.load_state_dict(
        torch.load(
            "models/resnet18_brain_tumor.pth",
            map_location=device
        )
    )

    model = model.to(device)
    
    model.eval()

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted_classes = torch.max(outputs, dim=1)

            all_predictions.append(predicted_classes)

            all_labels.append(labels)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)

    precision = precision_score(
        all_labels,
        all_predictions,
        average = "weighted"
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average = "weighted"
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average = "weighted"
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("Confusion Matrix:", matrix)

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_predictions))

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        yticklabels=["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=300)
            

if __name__ == "__main__":
    evaluate_model()



