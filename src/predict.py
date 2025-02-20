import torch
from torchvision import transforms, models
from PIL import Image
import argparse


def load_model(model_path):
    # Device configuration for Mac (MPS), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Initialize model architecture
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device


def predict_image(image_path, model, device):
    # Use the same transforms as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description='Predict handedness from image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model_path', type=str, default='handedness_model.pth',
                        help='Path to the trained model weights')
    args = parser.parse_args()

    # Load model
    model, device = load_model(args.model_path)

    # Make prediction
    predicted_class, confidence = predict_image(args.image_path, model, device)

    # Print results
    handedness = "Left-handed" if predicted_class == 0 else "Right-handed"
    print(f"Prediction: {handedness}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == '__main__':
    main()