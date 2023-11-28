from transformers import AutoProcessor, BlipForConditionalGeneration
from torch.utils.data import DataLoader

import csv
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from image_caption_dataset import ImageCaptioningDataset


# Initialize processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Assuming 'dataset' is already defined or loaded
# Create an instance of your custom dataset
csv_file_path = "/Users/nero/Desktop/CaptionThis/data_sets/downloaded_images.csv"
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])
train_dataset = ImageCaptioningDataset(csv_file_path, transform, processor)

# Create a DataLoader for the training dataset
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

# Training setup (optimizer, device setup)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Optional, if you want to use a scheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Training loop
TOTAL_EPOCHS = 10
for epoch in range(TOTAL_EPOCHS):
    print("Epoch:", epoch)
    model.train()

    total_loss = 0.0
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()  # Reset gradients

        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss

        if loss is not None:
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Batch {idx} Loss: {loss.item()}")
        else:
            print(f"Batch {idx} Loss: None")


    scheduler.step()  # Adjust the learning rate

    # Save the model after each epoch (optional but recommended)
    torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')

    # Print average loss
    avg_loss = total_loss / len(train_dataloader)
    print(f"Average Loss for Epoch {epoch}: {avg_loss}")
