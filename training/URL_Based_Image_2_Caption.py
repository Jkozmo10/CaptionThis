# Load and preprocess the image from URL with user input and timeout
from IPython.display import display, clear_output
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

DELAY_TIME = 3
DESIRED_WIDTH = 300
DESIRED_HEIGHT = 300
TIMEOUT_TIME = 5

# Define the device (cuda for GPU or cpu for CPU), GPU is better
device = "cuda" if torch.cuda.is_available() else "cpu"

while True:
  image_url = input("Enter the URL to a JPEG image: ")

  try:
      # A 5-second timeout to reduce system resource usage
      response = requests.get(image_url, timeout=TIMEOUT_TIME)
      response.raise_for_status()

      # Validate Image is a JPEG, but other formats should be allowed
      img = Image.open(BytesIO(response.content))
      if img.format != "JPEG":
        print("Error: The image is not in JPEG format.")
      else:
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
        print("\nGenerated Caption:", generated_caption)
        display(img)
        time.sleep(DELAY_TIME)
        # Clear the output to remove the image
        clear_output(wait=True)

        # Prompt the user for input
        user_input = input("Do you want to continue? (y/n): ").strip().lower()
        print()

        # Check if the user wants to continue or exit the loop
        if user_input == 'n':
            break

  except requests.exceptions.RequestException as e:
      print("Error:", str(e))
  except Exception as e:
      print("An error occurred:", str(e))

