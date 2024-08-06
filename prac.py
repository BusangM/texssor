# from transformers import BertTokenizer, BertModel
# import torch

# # Load pre-trained model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Input text
# text = "I love NLP"

# # Tokenize the input text
# inputs = tokenizer(text, return_tensors="pt")
# print("Tokens:", inputs['input_ids'])

# # Get the embeddings using the model
# with torch.no_grad():
#     outputs = model(**inputs)

# # The last hidden state is typically used as the embeddings
# embeddings = outputs.last_hidden_state
# print("Embeddings:", embeddings)

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "The cat sat on the mat.",
    "The dog sat on the log."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert the TF-IDF matrix to a DataFrame for better readability
import pandas as pd
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

print(tfidf_df)

"""
        cat       dog       log  ...        on       sat       the
0  0.445548  0.000000  0.000000  ...  0.317011  0.317011  0.634021
1  0.000000  0.445548  0.445548  ...  0.317011  0.317011  0.634021

Columns
Terms (Words): Each column corresponds to a unique term from the documents. In this case, the terms are "cat", "dog", "log", "on", "sat", and "the".
Rows
Documents: Each row corresponds to one of the input documents. In this case, there are two documents:
Document 0: "The cat sat on the mat."
Document 1: "The dog sat on the log."
Values
TF-IDF Scores: The values in the DataFrame are the TF-IDF scores for each term in each document. These scores represent the importance of each term in the document relative to the entire corpus.
Example Interpretation
Row 0 (Document 0):

The term "cat" has a TF-IDF score of 0.445548.
The term "dog" has a TF-IDF score of 0.000000 (it does not appear in Document 0).
The term "the" has a TF-IDF score of 0.634021, indicating it is a common term in Document 0 but also appears in Document 1, reducing its uniqueness.
Row 1 (Document 1):

The term "dog" has a TF-IDF score of 0.445548.
The term "cat" has a TF-IDF score of 0.000000 (it does not appear in Document 1).
The term "the" has a TF-IDF score of 0.634021, similar to Document 0.
Summary
Higher TF-IDF Score: Indicates that the term is more important in the document and less common across the entire corpus.
Lower TF-IDF Score: Indicates that the term is either less important in the document or more common across the entire corpus.
This TF-IDF matrix can be used as input features for machine learning models to perform tasks such as text classification, clustering, or similarity analysis.
"""