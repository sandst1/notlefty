import json
import requests
from pathlib import Path

# Create directories if they don't exist
Path("data/left").mkdir(parents=True, exist_ok=True)
Path("data/right").mkdir(parents=True, exist_ok=True)

def download_images(urls, folder):
    for i, url in enumerate(urls):
        try:
            # Extract filename from URL
            filename = url.split('/')[-1]

            # Download image
            response = requests.get(url)
            response.raise_for_status()

            # Save image
            save_path = f"data/{folder}/{filename}"
            with open(save_path, 'wb') as f:
                f.write(response.content)

            print(f"Downloaded {i+1}/{len(urls)}: {filename}")

        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")


# Load URLs from JSON files
with open('data/lefty-urls.json') as f:
    lefty_data = json.load(f)
    lefty_urls = lefty_data['urls'] if isinstance(lefty_data, dict) else lefty_data

with open('data/righty-urls.json') as f:
    righty_data = json.load(f)
    righty_urls = righty_data['urls']

# Download images
print("Downloading left-handed guitar images...")
download_images(lefty_urls, "left")

print("\nDownloading right-handed guitar images...")
download_images(righty_urls, "right")

print("\nDownload complete!")
