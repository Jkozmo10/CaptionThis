import torch
from torch.utils.data import DataLoader, random_split
from transformers import BlipProcessor, BlipForConditionalGeneration
from image_caption_dataset import ImageCaptionDataset, get_transform
from tqdm import tqdm
import logging

def get_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Dataset setup
        csv_file_path = '/path/to/sorted_output.csv'
        transform = get_transform()
        dataset = ImageCaptionDataset(csv_file_path, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Model and Training Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        num_epochs = 50

        # Training Loop
        model.train()
        for epoch in range(num_epochs):
            logging.info(f"Epoch: {epoch+1}/{num_epochs}")
            for images, captions in tqdm(train_dataloader):
                try:
                    images, captions = images.to(device), captions.to(device)
                    inputs = processor(images=images, text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
                    outputs = model(**inputs)

                    optimizer.zero_grad()
                    if outputs.loss is not None:
                        outputs.loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                except Exception as e:
                    logging.error(f"Error during training: {e}")

            scheduler.step()

            # Save Checkpoint
            torch.save(model.state_dict(), f'trained_blip_model_epoch_{epoch+1}.pth')

        logging.info("Training complete.")

    except Exception as e:
        logging.error(f"Failed to complete training: {e}")

if __name__ == "__main__":
    main()

