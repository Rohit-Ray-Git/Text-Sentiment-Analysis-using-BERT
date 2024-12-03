# **BERT-Based Sentiment Analysis**

## Overview
This project implements sentiment analysis using **BERT (Bidirectional Encoder Representations from Transformers)**, a cutting-edge model for natural language understanding. It classifies text inputs as either **Positive** or **Negative** based on their sentiment. The application is powered by a pre-trained BERT model fine-tuned on sentiment data, offering robust, real-time sentiment predictions.

The project includes a sleek and interactive user interface built with **Streamlit**, enabling users to input text and receive sentiment analysis predictions instantly. Whether for research, business, or personal use, this tool provides an efficient, state-of-the-art solution for analyzing the sentiment of textual data.

---

## **Key Features**
- **Real-time Sentiment Analysis**: Predict the sentiment (positive/negative) of text input immediately.
- **BERT Transformer Model**: Utilizes the BERT model, fine-tuned specifically for sentiment classification tasks.
- **Interactive UI**: Built using **Streamlit**, the app provides an intuitive and professional user interface for seamless interaction with the model.
- **Lightweight and Fast**: Optimized for speed and performance, ensuring rapid text analysis even with long sentences.

---

## **Technologies Used**
- **BERT (Transformers)**: A pre-trained NLP model for high-quality text representation.
- **Streamlit**: A Python library for building fast and interactive web applications, ideal for data science and machine learning projects.
- **Hugging Face Transformers**: A library for transformer models, making it easy to use pre-trained models like BERT.
- **PyTorch**: A deep learning framework for model inference.
- **Python 3.x**: The programming language used for all development.

---

## **Installation and Setup**

### 1. **Clone the Repository**:
First, clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/bert-sentiment-analysis.git
cd bert-sentiment-analysis
```

### 2. **Create a Virtual Environment**:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. **Install Project Dependencies**:
Install all required dependencies via pip:
```bash
pip install -r requirements.txt
```

### 4. **Download Pre-trained BERT Model**:
Ensure the pre-trained model and tokenizer files are located in the folder `bert_imdb_model/` (or your preferred folder name). The folder should contain:

- `config.json`
- `model.safetensors`
- `special_tokens_map.json`
- `tokenizer_config.json`
- `vocab.txt`

## **Usage**
### 1. **Run the Streamlit App**:
Once the environment is set up, run the Streamlit application:
```
streamlit run app.py
```

### 2. **Text Input and Prediction**:
- Enter any text in the input field (e.g., "I love this movie!").
- Click the Predict Sentiment button.
- The app will display whether the sentiment of the text is Positive or Negative.


