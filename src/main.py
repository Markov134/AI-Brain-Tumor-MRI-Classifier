from database import *
from preprocessing import *
from model import *

def main():
    metadata = load_metadata()
    train_loader, test_loader = create_dataloaders(metadata)

    print(len(train_loader))
    print(len(test_loader))

    model = create_model()
    print(model)
    

if __name__ == "__main__":
    main()