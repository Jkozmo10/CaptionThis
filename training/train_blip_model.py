# Import necessary libraries
import os
import requests
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None, processor=None):
        self.data = []
        self.transform = transform
        self.processor = processor  # Add the processor as an attribute
        self.image_dir = '/Users/nero/Desktop/CaptionThis/data_sets/Images'
        try:
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header row
                self.data = [row for row in reader]
        except IOError as e:
            logging.error(f"Error reading CSV file: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_filename, caption = self.data[idx][1], self.data[idx][2]
        image_path = os.path.join(self.image_dir, image_filename)
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            caption = self.data[idx][2]  # Caption as a string
            return image, caption  # Return the caption as a string
        except (IOError, UnidentifiedImageError) as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None, None

# Define the transformation applied to each image
def get_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])

# Define the main function for training
def main():
    try:
        # Dataset setup
        csv_file_path = '/Users/nero/Desktop/CaptionThis/data_sets/downloaded_images.csv'
        transform = get_transform()

        # Model and Training Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        dataset = ImageCaptionDataset(csv_file_path, transform=transform, processor=processor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=False)

        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        num_epochs = 10

        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            model.train()  # Ensure the model is in training mode
            total_loss = 0.0

            for batch_idx, (images, captions) in enumerate(train_dataloader):
                try:
                    images = images.to(device)
                    inputs = processor(images=images, text=list(captions), return_tensors="pt", padding=True, truncation=True).to(device)
                    outputs = model(**inputs)

                    loss = outputs.loss
                    if loss is not None:
                        total_loss += loss.item()
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        logging.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item()}")
                    else:
                        print("Loss is None")

                except AttributeError as e:
                    logging.error(f"Attribute error in batch {batch_idx}: {e}")
                    continue  # Skip this batch due to an error
                except Exception as e:
                    logging.error(f"Unexpected error during training at batch {batch_idx}: {e}")
                    continue  # Skip this batch due to an error

            avg_loss = total_loss / len(train_dataloader)
            logging.info(f"Epoch {epoch+1}/{num_epochs} completed with average loss: {avg_loss}")
            scheduler.step()  # Adjust the learning rate based on the scheduler
            torch.save(model.state_dict(), f'trained_blip_model_epoch_{epoch+1}.pth')

        logging.info("Training complete.")

    except Exception as e:
        logging.error(f"Failed to complete training: {e}")

if __name__ == "__main__":
    main()

