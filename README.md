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
$ source .venv/bin/activate
(.venv) $ pip install -r requirements.txt
```

### Verify installation 

```bash
pip list
```


