import argparse
import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import tensorflow as tf

DELAY_TIME = 3
DESIRED_WIDTH = 300
DESIRED_HEIGHT = 300
TIMEOUT_TIME = 5

# Define the device (cuda for GPU or cpu for CPU), GPU is better
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_image(image_path, output_csv):
    try:
        img = Image.open(image_path)
        img_format = img.format

        # Validate Image is a JPEG, but other formats should be allowed
        if img_format != "JPEG":
            print("Error: The image is not in JPEG format.")
            return

        # Resize the image to reduce system resource usage
        img = img.resize((DESIRED_WIDTH, DESIRED_HEIGHT), Image.LANCZOS)

        # Define the model and processor
        model_name = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        # Convert the image to torch format for the model and move to the same device as the model
        inputs = processor(images=img, return_tensors="pt").to(device)

        # Generate captions with model
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)

        # Decode and print the generated caption and display the image
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_caption = generated_caption.capitalize() + '.'

        # Save the caption and image path to the CSV file
        with open(output_csv, 'a') as file:
            file.write(f"{generated_caption},{image_path}\n")

    except Exception as e:
        print("An error occurred:", str(e))

def main():
    parser = argparse.ArgumentParser(description="Process a folder of images and generate a CSV with caption-image pair names.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input folder path containing images.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output CSV file path.")
    args = parser.parse_args()

    input_folder = args.input
    output_csv = args.output

    if not os.path.exists(input_folder):
        print("Error: Input folder does not exist.")
        return

    if not os.path.isdir(input_folder):
        print("Error: Input is not a directory.")
        return

    image_files = os.listdir(input_folder)

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        process_image(image_path, output_csv)

if __name__ == "__main__":
    main()
