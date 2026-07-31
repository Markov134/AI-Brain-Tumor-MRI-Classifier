import torch.nn as nn
import torchvision.models as models

def create_model():
    """Create a ResNet-18 model for brain tumor classification."""
    
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    input_features = model.fc.in_features

    model.fc = nn.Linear(input_features, 4)

    return model