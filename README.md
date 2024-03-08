# Fake News Detection using BERT-based Transfer Learning

## Overview
Fake News Detection is a project aimed at identifying fake news headlines using transfer learning with BERT (Bidirectional Encoder Representations from Transformers), a powerful natural language processing (NLP) model. The project includes training a custom neural network architecture, developing a user-friendly web application using Streamlit, and deploying the model for real-time predictions.

## Key Features
- Preprocessing and tokenization of news headlines dataset.
- Definition of a custom BERT-based neural network architecture for classification.
- Training the model using AdamW optimizer and cross-entropy loss function.
- Development of a Streamlit web application for real-time predictions.
- Deployment of the project locally for easy access.

## Dependencies
- Python 3.x
- PyTorch
- Transformers library from Hugging Face
- Streamlit

## Installation
1. Clone the repository:
git clone https://github.com/OUSSAMAAKAABOUR/fake-news-detection.git

2. Install the required dependencies:
pip install -r requirements.txt


## Usage
1. Navigate to the project directory.
2. Run the Streamlit app:

streamlit run fake_news_detector_app.py

3. Access the web application in your browser and input news headlines for prediction.

## Dataset
The project uses a labeled dataset of news headlines, containing examples of both true and fake news.

## Model Architecture
The custom neural network architecture includes fine-tuning BERT layers followed by additional fully connected layers for classification.

## Deployment
The project is deployed locally, allowing users to access the fake news detection functionality through the Streamlit web application.

## Credits
- The project utilizes the Transformers library from Hugging Face for BERT-based NLP tasks.
- Inspired by research in the field of natural language processing and fake news detection.


## Author
AKAABOUR OUSSAMA
