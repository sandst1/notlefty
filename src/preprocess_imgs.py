import os
from PIL import Image
import argparse


def preprocess_images(folder_path):
    """
    Read images from folder, resize them to 224x224, and save with 'resized_' prefix

    Args:
        folder_path (str): Path to folder containing images

    Returns:
        list: List of paths to resized images
    """
    processed_image_paths = []

    # Iterate through all files in folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            # Open image
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)

            # Resize to 224x224
            img_resized = img.resize((224, 224))

            # Generate output filename and save
            name, ext = os.path.splitext(filename)
            output_filename = f"resized_{name}.jpg"
            output_path = os.path.join(folder_path, output_filename)

            img_resized.save(output_path, "JPEG")
            processed_image_paths.append(output_path)

    return processed_image_paths


def main():
    """Process images in data/left and data/right directories"""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Preprocess images for left/right hand classification'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Base directory containing left and right subdirectories'
    )

    args = parser.parse_args()

    # Process both directories
    left_dir = os.path.join(args.data_dir, 'left')
    right_dir = os.path.join(args.data_dir, 'right')

    for directory in [left_dir, right_dir]:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)

        print(f"Processing images in {directory}...")
        processed = preprocess_images(directory)
        print(f"Processed {len(processed)} images in {directory}")


if __name__ == '__main__':
    main()

