# image_caption_dataset.py

import csv
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.data.append(row)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            image_path, caption, _ = self.data[idx]
            image_path = "/Users/nero/Desktop/CaptionThis/data_sets/Images/" + image_path
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, caption
        except UnidentifiedImageError:
            return None, None

# Example transformation, can be customized as needed
def get_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
