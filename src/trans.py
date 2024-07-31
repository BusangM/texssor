import transformers

# Here we are loading the pre-trained distil-base-uncased model and tokenizer

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased') #This is the tokenizer for DistilBERT

model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased') #This is the model for DistilBERT

# Now we can use the tokenizer to tokenize the input text and then use the model to get the embeddings

sample_text = "This is a sample text to test the tokenizer and model."

inputs = tokenizer(sample_text, return_tensors="pt") # This will tokenize the input text and return the input tensors
# pt argument specifies that the output should be returned as PyTorch tensors.

outputs = model(**inputs) # This will return the embeddings

print(outputs)