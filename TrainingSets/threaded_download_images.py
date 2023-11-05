import os
import requests
import csv
from concurrent.futures import ThreadPoolExecutor

# Constants
output_directory = "Images"
os.makedirs(output_directory, exist_ok=True)

tsv_files = [
    "Image_Labels_Subset_Train_GCC-Labels-training.tsv",
    "Train_GCC-training.tsv",
    "Validation_GCC-1.1.0-Validation.tsv"
]

SUCCESS_STATUS_CODE = 200
TIMEOUT_TIME = 5
NUMBER_OF_THREADS = 8

def download_image(idx, caption, image_url):
    try:
        response = requests.get(image_url, timeout=TIMEOUT_TIME)
        if response.status_code == SUCCESS_STATUS_CODE:
            image_filename = f"{idx + 1}.jpg"
            image_path = os.path.join(output_directory, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(response.content)
            print(f"Downloaded: {image_filename}")
        else:
            print(f"Failed to download: {image_url}")
    except Exception as e:
        print(f"Error downloading image {idx + 1}: {str(e)}")

if __name__ == "__main__":
    with open(tsv_files[1], "r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        tasks = []
        for idx, (caption, image_url) in enumerate(tsv_reader):
            tasks.append((idx, caption, image_url))

        with ThreadPoolExecutor(max_workers=NUMBER_OF_THREADS) as executor:
            for task in tasks:
                executor.submit(download_image, *task)

    print("Download complete.")
