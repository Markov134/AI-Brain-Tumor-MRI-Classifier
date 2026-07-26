from database import *
from preprocessing import *

def main():
    metadata = load_metadata()
    train_loader, test_loader = create_dataloaders(metadata)

    print(len(train_loader))
    print(len(test_loader))

if __name__ == "__main__":
    main()