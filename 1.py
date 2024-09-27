from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-es"  # or any other model you need
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./tokenizer")
