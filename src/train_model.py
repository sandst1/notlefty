import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np


class HandednessDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def get_data_paths():
    left_dir = 'data/left'
    right_dir = 'data/right'

    left_images = [os.path.join(left_dir, f) for f in os.listdir(left_dir) if f.startswith('resized_')]
    right_images = [os.path.join(right_dir, f) for f in os.listdir(right_dir) if f.startswith('resized_')]

    all_paths = left_images + right_images
    labels = [0] * len(left_images) + [1] * len(right_images)

    return all_paths, labels


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Device configuration for Mac (MPS), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Get data paths and create train/val split
    image_paths, labels = get_data_paths()
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = HandednessDataset(X_train, y_train, transform=transform)
    val_dataset = HandednessDataset(X_val, y_val, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)

    # Modify the classifier for binary classification
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0.0
    print("\nStarting training...")
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    print("-" * 80)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Add batch progress tracking
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Print batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] '
                      f'Batch [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                # Calculate validation loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate epoch statistics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Training Loss: {avg_train_loss:.4f} | Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'handedness_model.pth')
            print(f'New best validation accuracy: {val_acc:.2f}%! Saved model.')
        print("-" * 80)


if __name__ == '__main__':
    main()
