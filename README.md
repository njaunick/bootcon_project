# BOOTCON

## Members
- Symia Woodson
- Wanda Knight
- Nickson Njau

# Emergency Call Response 

## Classification

### Project Overview
- This project implemented a machine learning model to classify emergency call responses using BERT (bert-base-uncased). It distinguished between caller and assistant responses in 911 call transcripts to automate the categorization of emergency call conversations.

# Table of Contents
1. Features
2. Requirements
3. Dataset
4. Model Architecture
5. Training Process
6. Usage
7. Performance
8. Future Improvements

## Features
- Utilized the "911-call-transcripts" dataset from Kaggle
- Custom PyTorch Dataset for efficient data handling
- BERT-based text classification model
- GPU-accelerated training (with CPU fallback)
- Automated text preprocessing and tokenization
- Train-validation split for model evaluation
- Multi-epoch training with validation
- Learning rate scheduling with warmup
- Performance tracking (loss and accuracy)

## Requirements
- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- Scikit-learn
- Datasets (Hugging Face)
- KaggleHub


## Install required packages:
- pip install torch transformers pandas scikit-learn datasets kagglehub

## Dataset
- The project used the "911-call-transcripts" dataset from Kaggle, loaded using the Hugging Face datasets library. 
- The dataset is processed to create a binary classification task:
	- Label 0: Emergency Calls
	- Label 1: Non-Emergency Calls
## Model Architecture
- Base model: BERT (bert-base-uncased)
- Additional classification layer for binary classification
- Maximum sequence length: 512 tokens

## Training Process
### Data Preparation:
- Load dataset using load_dataset from Hugging Face
- Preprocess data: expand messages, create labels, remove null entries
- Split into training and validation sets using train_test_split

### Model Setup:
- Initialized BERT tokenizer and model
- Created custom EmergencyCallDataset class
- Set up data loaders with batch size of 8

### Training Loop:
- 10 epochs of training
- AdamW optimizer with learning rate of 2e-5
- Linear learning rate scheduler with warmup
- Validation after each epoch

### Performance Tracking:
- Calculate and print validation loss and accuracy after each epoch

## Performance
- The model's performance was measured by:
	- Validation Loss: Calculated after each epoch - The best performing model had a Validation loss of 0.2095
	- Accuracy: Percentage of correctly classified samples in the validation set - The best performing model had a Accuracy of 93.06%

## Future Improvements
- Experiment with different transformer architectures (e.g., RoBERTa, DistilBERT)
- Analyze model predictions to identify patterns in emergency call responses
