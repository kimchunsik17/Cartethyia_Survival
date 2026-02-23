import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Configuration ---
DATA_DIR = "assets/spell_data"
MODEL_SAVE_PATH = "spell_model.pth"
BATCH_SIZE = 8 # Small batch size because dataset is very small
EPOCHS = 100
IMG_SIZE = 64
NUM_CLASSES = 6

# --- Model Definition ---
class SpellCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SpellCNN, self).__init__()
        # Input: 1 channel (Grayscale) x 64 x 64
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 16 x 32 x 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 32 x 16 x 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> 64 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # prevent overfitting on small data
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Augmentation to help with tiny dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    print(f"Classes found: {dataset.class_to_idx}")
    print(f"Total dataset size: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Init model, loss, optimizer
    model = SpellCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

    print("Training finished!")
    
    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Output classes list so game can use it
    classes = dataset.classes
    print(f"Class mapping: {classes}")

if __name__ == "__main__":
    main()
