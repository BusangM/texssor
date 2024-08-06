import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def encode_text(input_file, output_file):

    data = pd.read_csv(input_file) # Load the preprocessed dataset

    texts = data['preprocessed_text'].tolist() # Extract the preprocessed text

    toVectorize = TfidfVectorizer() # Initialises the TF-IDF vectorizer()
    #This transforms text data into numerical vectors that can be used by machine learning algorithms
    tfidf_matrix = toVectorize.fit_transform(texts)

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=toVectorize.get_feature_names_out()) # Convert the TF-IDF matrix to a DataFrame for better readability
    
    tfidf_df.to_csv(output_file, index=False) # Save the encoded data
    
    print(tfidf_df) # Print the TF-IDF DataFrame
    
    """
    The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is a tool used to convert a collection of raw text documents into a matrix of TF-IDF features.
    It is commonly used in natural language processing (NLP) and text mining to transform text data into numerical vectors that can be used by machine learning algorithms.

    How TF-IDF Works:
    - Term Frequency (TF): Measures how frequently a term appears in a document. The more frequently a term appears, the higher its TF value. [ \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d} ]
    
    - Inverse Document Frequency (IDF): Measures how important a term is by considering how common or rare it is across all documents. The more documents a term appears in, the lower its IDF value. [ \text{IDF}(t) = \log \left( \frac{\text{Total number of documents}}{\text{Number of documents containing term } t} \right) ]
    
    - TF-IDF: Combines the TF and IDF values to give a weight to each term in each document. This weight reflects the importance of the term in the document relative to the entire corpus. [ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) ]
    
    Purpose of TF-IDF Vectorizer:
    - Feature Extraction: Converts text data into numerical vectors that can be used as features for machine learning models.
    - Importance Weighting: Assigns higher weights to terms that are more important (i.e., terms that are frequent in a document but not common across all documents).
    - Dimensionality Reduction: Helps in reducing the dimensionality of the text data by focusing on the most important terms.
    
    """

if __name__ == "__main__":
    input_file = 'data/preprocessed_responses.csv'
    output_file = 'data/tfidf_encoded_responses.csv'
    encode_text(input_file, output_file)