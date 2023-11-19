import os
import requests
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
OUTPUT_DIRECTORY = "Images"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

TSV_FILE = "Train_GCC-training.tsv"
SUCCESS_STATUS_CODE = 200
TIMEOUT_TIME = 5
NUMBER_OF_THREADS = 10
MAXIMUM_NUMBER_OF_IMAGES = 100000

def download_image(idx, caption, image_url):
    try:
        response = requests.get(image_url, timeout=TIMEOUT_TIME)
        if response.status_code == SUCCESS_STATUS_CODE:
            image_filename = f"{idx}.jpg"
            image_path = os.path.join(OUTPUT_DIRECTORY, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(response.content)
            logging.info(f"Downloaded: {image_filename}")
            return image_filename, caption, image_url
    except Exception as e:
        logging.error(f"Error downloading image {idx}: {str(e)}")
    return None

def process_images():
    with ThreadPoolExecutor(max_workers=NUMBER_OF_THREADS) as executor:
        future_to_image = {}
        with open(TSV_FILE, "r", encoding="utf-8") as file:
            tsv_reader = csv.reader(file, delimiter="\t")
            for idx, (caption, image_url) in enumerate(tsv_reader):
                if idx >= MAXIMUM_NUMBER_OF_IMAGES:
                    break
                future = executor.submit(download_image, idx, caption, image_url)
                future_to_image[future] = idx

        downloaded_images_info = []
        for future in as_completed(future_to_image):
            result = future.result()
            if result:
                downloaded_images_info.append(result)

    return downloaded_images_info

if __name__ == "__main__":
    downloaded_images_info = process_images()

    downloaded_images_csv = os.path.join(OUTPUT_DIRECTORY, "downloaded_images.csv")
    with open(downloaded_images_csv, "w", encoding="utf-8") as file:
        csv_writer = csv.writer(file, delimiter=",")
        csv_writer.writerow(["Image Filename", "Caption", "Image URL"])
        for info in downloaded_images_info:
            csv_writer.writerow(info)

    logging.info("Download complete.")
