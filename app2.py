import streamlit as st
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
    1: "😊 Positive",
    0: "😞 Negative"
}

# Streamlit UI with enhanced styling
st.set_page_config(page_title="Sentiment Analysis", page_icon="📝", layout="centered")

# Title with some emoji
st.title("Sentiment Analysis with BERT 🧠")

# Add a description with emojis
st.markdown("""
    This is a simple sentiment analysis app built using the **BERT model**.
    Enter some text below, and the model will predict whether it has a **positive** or **negative** sentiment. 
    🌟
""")

# Create a centered layout for the input
st.markdown("""
    <style>
    .stTextArea>div>div>textarea {
        width: 100%;
        height: 150px;
        font-size: 18px;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Input text area in the center
text_input = st.text_area("Type your text here:", "", height=200)

# Create some space
st.write("\n" * 2)

# Predict button with an emoji
if st.button("🔮 Predict Sentiment"):
    if text_input:
        # Get the sentiment prediction
        numeric_prediction = predict_sentiment(text_input)
        sentiment = sentiment_map[numeric_prediction]
        
        # Show the result with a visual style
        st.markdown(f"""
        <h3 style='text-align: center; color: #4CAF50;'>{sentiment}</h3>
        """, unsafe_allow_html=True)
        
        # Add some additional information
        st.markdown("### 🌍 Explore the text sentiment above!")
    else:
        st.warning("Please enter some text to analyze.")

# Add a footer to make it more professional
st.markdown("""
    ---
    Created with ❤️ by **Rohit** | Powered by [Streamlit](https://streamlit.io) & [Hugging Face Transformers](https://huggingface.co)
""", unsafe_allow_html=True)
