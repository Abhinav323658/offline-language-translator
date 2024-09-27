from transformers import MarianMTModel, MarianTokenizer

def load_model_and_tokenizer(local_model_path, local_tokenizer_path):
    tokenizer = MarianTokenizer.from_pretrained(local_tokenizer_path)
    model = MarianMTModel.from_pretrained(local_model_path)
    return model, tokenizer

# Load model and tokenizer from local paths
model_path = "./model"
tokenizer_path = "./tokenizer"
model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
