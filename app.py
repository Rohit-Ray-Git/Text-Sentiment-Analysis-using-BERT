from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained BERT model and tokenizer
def load_bert_model():
    model = BertForSequenceClassification.from_pretrained('bert_imdb_model')
    tokenizer = BertTokenizer.from_pretrained('bert_imdb_model')
    return model, tokenizer

# Initialize the model and tokenizer when the app starts
model, tokenizer = load_bert_model()

# Model prediction logic using BERT
def predict_sentiment(text: str) -> int:
    # Tokenize and encode the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Perform inference and get the model's output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class (0 or 1)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class  # Return 0 (Negative) or 1 (Positive)

# Map numeric predictions to human-readable sentiments
sentiment_map = {
    1: "Positive",
    0: "Negative"
}

# Define the input schema
class TextRequest(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict/")
def predict(request: TextRequest):
    # Get prediction
    numeric_prediction = predict_sentiment(request.text)
    sentiment = sentiment_map[numeric_prediction]
    
    # Return more descriptive output
    return {
        "prediction": numeric_prediction,
        "sentiment": sentiment
    }
