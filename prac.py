from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Input text
text = "I love NLP"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")
print("Tokens:", inputs['input_ids'])

# Get the embeddings using the model
with torch.no_grad():
    outputs = model(**inputs)

# The last hidden state is typically used as the embeddings
embeddings = outputs.last_hidden_state
print("Embeddings:", embeddings)
