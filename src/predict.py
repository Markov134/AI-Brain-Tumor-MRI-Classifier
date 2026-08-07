import torch
from PIL import Image

from preprocessing import create_transforms
from model import create_model

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

def predict_image(image_path):
    """Predict the class of a single MRI image."""

    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    print(f"Using device: {device}")

    model = create_model()

    model.load_state_dict(
        torch.load(
            "models/resnet18_brain_tumor.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    transform = create_transforms()

    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    # Adds a batch dimension because the model expects input in the form:
    # (batch_size, channels, height, width).
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        # Converts raw model scores into probabilities that sum to 1.
        probabilities = torch.softmax(outputs, dim=1)

        # Selects the class with the highest probability as the prediction.
        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    print(f"\nPrediction: {CLASS_NAMES[predicted_class.item()]}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
