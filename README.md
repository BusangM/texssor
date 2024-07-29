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





