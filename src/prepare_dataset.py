import pandas as pd
import re


def preprocess_text(text):
    #Note to self: Some models might require to remove punctuation and special characters. Didn't include punctuation removal here for simplicity.
    return ' '.join(text.lower().split()) # Cleaning up data, removing whitespaces, converting to lowercase 


def prepare_dataset(input_file, output_file):

    data = pd.read_csv(input_file)
    data['preprocessed_text'] = data['response'].apply(preprocess_text)
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = 'data/responses.csv'
    output_file = 'data/preprocessed_responses.csv'
    prepare_dataset(input_file, output_file)