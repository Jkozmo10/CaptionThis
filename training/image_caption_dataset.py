import csv
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import os

class ImageCaptioningDataset(Dataset):
    def __init__(self, csv_file, transform, processor):
        self.data = []
        self.transform = transform
        self.processor = processor
        self.image_dir = '/Users/nero/Desktop/CaptionThis/data_sets/Images'  # Update with the path to your image directory

        # Read the CSV file and store the data
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["Image Filename"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Use processor to process the image and caption
        encoding = self.processor(images=image, text=item["Caption"], padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return encoding

