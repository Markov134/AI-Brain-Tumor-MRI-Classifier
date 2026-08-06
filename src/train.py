import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

from preprocessing import load_metadata, create_dataloaders
from model import create_model

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12,6)

def train_model():
    """Trains the ResNet-18 model using MRI image data."""

    # Checks if CUDA is available. Uses GPU for acceleration if possible,
    # otherwise uses CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    metadata = load_metadata()
    train_loader, _ = create_dataloaders(metadata)

    model = create_model()
    model = model.to(device)

    # Computes the difference between predicted classes and true labels.
    criterion = nn.CrossEntropyLoss()

    # Updates model weights using gradients from backpropagation.
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    num_epochs = 5

    history = {
        "loss": []
    }

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Reset gradients from the previous training step.
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        history["loss"].append(average_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}"
        )

    plt.plot(
        range(1, num_epochs + 1),
        history["loss"],
        marker="o"
    )
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig("images/training_loss.png", dpi=300)

    # Saves trained model weights for future evaluation and predictions.
    torch.save(model.state_dict(), "models/resnet18_brain_tumor.pth")

if __name__ == "__main__":
    train_model()



