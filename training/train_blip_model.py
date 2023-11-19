# train_blip_model.py

import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from image_caption_dataset import ImageCaptionDataset, get_transform

def main():
    # Set up dataset and dataloader
    csv_file_path = 'path_to_your_csv_file.csv'  # Replace with your CSV file path
    transform = get_transform()
    dataset = ImageCaptionDataset(csv_file_path, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the BLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    num_epochs = 10  # Define the number of epochs

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for idx, (images, captions) in enumerate(train_dataloader):
            images = images.to(device)

            # Process images and captions for BLIP model
            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)

            loss = outputs.loss
            print(f"Loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), 'trained_blip_model.pth')

if __name__ == "__main__":
    main()
