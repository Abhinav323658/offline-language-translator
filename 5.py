from transformers import MarianMTModel, MarianTokenizer
import os

# Function to download and save models locally
def download_and_save_model(model_name, save_dir):
    # Create directories if not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Download the model and tokenizer
    print(f"Downloading {model_name}...")
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    # Save the model and tokenizer locally
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Model {model_name} saved to {save_dir}")

# Specify the models you want to download
models_to_download = {
    "English to Spanish": "Helsinki-NLP/opus-mt-en-es",
    "English to French": "Helsinki-NLP/opus-mt-en-fr"
}

# Directory to save models
base_dir = "./offline_models"

# Download each model
for name, model_name in models_to_download.items():
    save_dir = os.path.join(base_dir, name.replace(" ", "_").lower())
    download_and_save_model(model_name, save_dir)

print("All models downloaded and saved.")
