import streamlit as st
import torch
import numpy as np
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn

# Load the tokenizer and model
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Define the BERT architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):  
        super(BERT_Arch, self).__init__()
        self.bert = bert   
        self.dropout = nn.Dropout(0.1)            # dropout layer
        self.relu =  nn.ReLU()                    # relu activation function
        self.fc1 = nn.Linear(768,512)             # dense layer 1
        self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
        
    def forward(self, sent_id, mask):           # define the forward pass  
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                           # output layer
        x = self.softmax(x)                       # apply softmax activation
        return x

# Load the model
model = BERT_Arch(bert)  # Assume 'bert' is defined elsewhere

# Load the model weights
model.load_state_dict(torch.load('c2_new_model_weights.pt', map_location=torch.device('cpu')))
model.eval()

# Define function to make predictions
def predict(tweet):
    # Tokenize and encode sequences
    MAX_LENGTH = 15
    tokens_unseen = tokenizer.batch_encode_plus(
        tweet,
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )
    unseen_seq = torch.tensor(tokens_unseen['input_ids'])
    unseen_mask = torch.tensor(tokens_unseen['attention_mask'])

    # Make predictions
    with torch.no_grad():
        preds = model(unseen_seq, unseen_mask)
        preds = preds.argmax(dim=1).cpu().numpy()

    return preds

# Streamlit app
def main():
    st.title("Fake News Detector")

    # Text area for user input
    tweet = st.text_area("Enter your news title:")

    # Button to trigger prediction
    if st.button("Check"):
        if tweet.strip() == "":
            st.warning("Please enter a news title.")
        else:
            # Make prediction
            predictions = predict([tweet])
            if predictions[0] == 0:
                st.error("This news title is likely to be fake.")
            else:
                st.info("This news title is likely to be true.")
                

# Run the app
if __name__ == "__main__":
    main()
