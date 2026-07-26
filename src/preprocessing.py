from torchvision.transforms import v2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from database import connect_db

LABEL_MAP = {
    'glioma': 0,
    'meningioma': 1,
    'notumor': 2,
    'pituitary': 3
}

def load_metadata():
    """Load MRI image metadata from the SQLite database."""
    connection = connect_db()
    cursor = connection.cursor()
    cursor.execute("SELECT filepath, diagnosis, split FROM mri_images;")
    metadata = cursor.fetchall()
    
    connection.close()
    return metadata

def create_transforms():
    """Create the image preprocessing pipeline."""
    return v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        # Convert image to float32 and scale pixel values from [0, 255] to [0, 1].
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean = (0.5, 0.5, 0.5), 
            std = (0.5, 0.5, 0.5)
        )
    ])

class MRIDataset(Dataset):
    """Custom PyTorch Dataset for loading MRI images and labels."""
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        filepath, diagnosis, _ = self.metadata[index]

        img = Image.open(filepath).convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        label = LABEL_MAP[diagnosis]
        
        return img, label

def create_dataloaders(metadata, batch_size=32):
    """Create PyTorch DataLoaders for the training and testing datasets."""
    train_metadata = [
        row for row in metadata
        if row[2].lower() == "training"
    ]

    test_metadata = [
        row for row in metadata
        if row[2].lower() == "testing"
    ]

    transform = create_transforms()

    train_dataset = MRIDataset(
        train_metadata,
        transform=transform
    )

    test_dataset = MRIDataset(
        test_metadata,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader