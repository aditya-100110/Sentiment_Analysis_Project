"# Sentiment_Analysis" 


# Sentiment Analysis of Reviews

This project is a simple web application for performing sentiment analysis on product reviews using the Naive Bayes algorithm. It is built with Python, scikit-learn, and Streamlit.

## Features

- Classifies reviews into Positive or Negative sentiment
- Trained using the Amazon Reviews dataset
- Easy-to-use Streamlit web interface

## How to Run

1. Clone the repository or download the project files.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model (if not already trained):

```bash
python train_model.py
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Files

- `app.py`: Streamlit frontend for sentiment prediction
- `train_model.py`: Script to train and save the Naive Bayes model
- `model.pkl`: Trained model file
- `vectorizer.pkl`: Vectorizer used for transforming review text
- `requirements.txt`: List of required Python packages
- `1429_1.csv`: Sample dataset used for training

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- streamlit
 
 ## Data set
 Amazon Reviews Data Set
 https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
