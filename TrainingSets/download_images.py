import os
import requests
import time
import csv

# Create a directory to store the downloaded images
output_directory = "Images"
os.makedirs(output_directory, exist_ok=True)

# Read the TSV file with captions and image URLs
tsv_files = [
    "Image_Labels_Subset_Train_GCC-Labels-training.tsv",
    "Train_GCC-training.tsv",
    "Validation_GCC-1.1.0-Validation.tsv"
]

SUCCESS_STATUS_CODE = 200
DELAY_TIME = 1

with open(tsv_files[1], "r", encoding="utf-8") as file:
    tsv_reader = csv.reader(file, delimiter="\t")
    for idx, (caption, image_url) in enumerate(tsv_reader):
        if idx < 10000:
            try:
                response = requests.get(image_url, timeout=5)
                if response.status_code == SUCCESS_STATUS_CODE:
                    # Generate a unique filename for each image based on the index
                    image_filename = f"{idx + 1}.jpg"
                    image_path = os.path.join(output_directory, image_filename)

                    # Save the image to the local directory
                    with open(image_path, "wb") as image_file:
                        image_file.write(response.content)
                    print(f"Downloaded: {image_filename}")
                else:
                    print(f"Failed to download: {image_url}")
            except Exception as e:
                print(f"Error downloading image {idx + 1}: {str(e)}")

            # Add a delay between requests to avoid overloading the server
            #time.sleep(DELAY_TIME)  # Adjust this delay as needed

print("Download complete.")

