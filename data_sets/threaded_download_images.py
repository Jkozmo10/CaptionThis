import os
import requests
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from PIL import Image
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(essage)s')

# Constants
OUTPUT_DIRECTORY = "Images"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

TSV_FILE = "Train_GCC-training.tsv"
SUCCESS_STATUS_CODE = 200
TIMEOUT_TIME = 5
NUMBER_OF_THREADS = 10
MAXIMUM_NUMBER_OF_IMAGES = 35000
OUTPUT_CSV_DIR = "."
OUTPUT_CSV = "downloaded_images.csv"
VALID_IMAGE_FORMATS = {'JPEG', 'JPG', 'PNG'}

def download_image(idx, caption, image_url):
    try:
        response = requests.get(image_url, timeout=TIMEOUT_TIME)
        if response.status_code == SUCCESS_STATUS_CODE:
            image_content = BytesIO(response.content)
            try:
                with Image.open(image_content) as img:
                    if img.format in VALID_IMAGE_FORMATS:
                        ext = '.' + img.format.lower()
                        image_filename = f"{idx}{ext}"
                        image_path = os.path.join(OUTPUT_DIRECTORY, image_filename)
                        img.save(image_path)
                        logging.info(f"Downloaded and saved: {image_filename}")
                        return idx, image_filename, caption, image_url
                    else:
                        logging.warning(f"Unsupported image format for URL: {image_url}")
                        return None
            except IOError:
                logging.warning(f"Invalid image data for URL: {image_url}")
                return None
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

    # Sort the list by idx in ascending order
    downloaded_images_info = sorted(downloaded_images_info, key=lambda x: x[0])

    downloaded_images_csv = os.path.join(OUTPUT_CSV_DIR, OUTPUT_CSV)
    with open(downloaded_images_csv, "w", encoding="utf-8") as file:
        csv_writer = csv.writer(file, delimiter=",")
        csv_writer.writerow(["Index", "Image Filename", "Caption", "Image URL"])
        for idx, filename, caption, url in downloaded_images_info:
            csv_writer.writerow([idx, filename, caption, url])

    logging.info("Download complete.")
