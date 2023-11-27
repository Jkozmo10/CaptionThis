import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import logging

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.transform = transform
        try:
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                self.data = [row for row in reader]
        except IOError as e:
            logging.error(f"Error reading CSV file: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption, _ = self.data[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, caption
        except (IOError, UnidentifiedImageError) as e:
            logging.error(f"Error loading image: {e}")
            return None, None

