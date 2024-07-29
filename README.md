### Texssor

# Automated Grading System

This project uses PyTorch and the Hugging Face `transformers` library to automate the grading of long text responses.

## Requirements:
- PyTorch: Building and training neural network models. (Optional)
- Transformers: Loading and fine-tuning pre-trained NLP models. (specifically- pipeline)
- Pandas: Data manipulation and analysis, handling spreadsheet data.
- Openpyxl: Reading and writing Excel files.
- Scikit-learn: Dataset splitting and model evaluation.


These tools work together to create an automated grading system that leverages deep learning models for text classification, ensuring high accuracy and efficient handling of large volumes of text data.

## Setup Instructions

### Step 1: Create a Virtual Environment

   ```bash
    python -m venv venv
   ```

### Step 2: Activate & Install requirements into virtualenv
```bash
source .venv/bin/activate

pip install -r requirements.txt
```

### Verify installation 

```bash
pip list
```

## Hugging Face Transformers

The Hugging Face Transformers library provides access to a wide range of pre-trained models for various natural language processing (NLP) tasks. It simplifies the process of using these models by offering easy-to-use APIs for model loading, tokenization, and fine-tuning.

### Key Features

- **Pre-trained Models**: Access to state-of-the-art models like BERT, GPT-3, and DistilBERT.
- **Tokenization**: Efficiently convert text into tokens that the models can process.
- **Fine-Tuning**: Customise pre-trained models for specific tasks with minimal data.
- **Model Inference**: Easily perform tasks such as text classification, translation, and summarisation.

### Why Use Hugging Face Transformers?

1. **Efficiency**: Pre-trained models save time and computational resources.
2. **Performance**: State-of-the-art models ensure high accuracy.
3. **Ease of Use**: Simple APIs make it easy to integrate into your projects.

### How It Will Be Used in This Project

In this project, we will use the [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) model from Hugging Face Transformers to create an automated grading system. This model will be fine-tuned to classify text data, ensuring high accuracy and efficient handling of large volumes of text.

## Documentation
For more information, visit the [Hugging Face Transformers documentation](https://huggingface.co/transformers/)

These tools work together to create an automated grading system that leverages deep learning models for text classification, ensuring high accuracy and efficient handling of large volumes of text data.



## Important Keywords Explained In Simple English

#### Tokenization: 

- Definition: The process of breaking down a string of text into smaller chunks called tokens.
- Purpose: To convert the raw text into manageable pieces for further processing.
- Example: For the sentence "I love NLP":
 
      Word-level tokenization: ["I", "love", "NLP"]

      Subword-level tokenization: ["I", "lo", "ve", "NL", "P"]
      
      Character-level tokenization: ["I", " ", "l", "o", "v", "e", " ", "N", "L", "P"]


#### Embedding:

- Definition: The process of converting tokens into dense numerical vectors that represent the tokens in a meaningful way.
- Purpose: To transform the tokens into a format that a machine learning model can understand and process, capturing semantic relationships.
- Example: After tokenization, the tokens ["I", "love", "NLP"] might be converted to numerical vectors like:
      
      "I": [0.1, 0.2, 0.3]
      "love": [0.4, 0.5, 0.6]
      "NLP": [0.7, 0.8, 0.9]

#### Tensor

Definition:

A tensor is a generalisation of scalars, vectors, and matrices to potentially higher dimensions.
It can be thought of as an n-dimensional array.
   
   Dimensions:

      - Scalar: A single number (0-dimensional tensor). e.g 7

      - Vector: A 1-dimensional array of numbers. e.g [1,2,3]

      - Matrix: A 2-dimensional array of numbers. e.g
      |1 2|
      |3 4|
      - Higher-order Tensors: Arrays with 3 or more dimensions. 

      - 3D Tensor: An array of matrices, like a cube of numbers.

How they are related? 

After tokenizing a sentence, we often convert the tokens into numerical representations (token IDs).
These token IDs can be represented as tensors to be processed by neural networks.

When we use a model to generate embeddings, the output is typically a tensor.
For example, if we tokenize a sentence and use a model to get embeddings, the embeddings are stored in a tensor where each token has its corresponding embedding vector.

In conclusion, they are used to store token IDs and embeddings, enabling efficient processing by neural networks. Tensors are essential for handling the multi-dimensional data involved in deep learning tasks.

#### Neural Network

Neural Networks are computational models inspired by the human brain's structure and function. They are designed to recognise patterns and make predictions based on data. Neural networks consist of layers of interconnected nodes, or neurons, where each connection has an associated weight.