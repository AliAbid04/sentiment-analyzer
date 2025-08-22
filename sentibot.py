import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import render_template, request, Flask

# Initialize Flask application
app = Flask(__name__)

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# Load the pre-trained sentiment analysis model
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer used for text transformation
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to clean and preprocess input text
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters and punctuation
    # Remove stopwords (common words like "the", "is", "and" that do not add much meaning)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))  
    return text

# Function to predict sentiment of a given text input
def predict_sentiment(text):
    cleaned_text = clean_text(text)  # Preprocess the text
    transformed_text = vectorizer.transform([cleaned_text])  # Convert text into numerical features using TF-IDF
    prediction = model.predict(transformed_text)[0]  # Get the predicted sentiment
    return "Positive" if prediction == 2 else "Negative"  # Map numerical prediction to label

# Define the main route (homepage) with both GET and POST methods
@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""  # Default result is empty
    if request.method == 'POST':  # If a form is submitted
        review = request.form.get('review')  # Get user input from the form
        if review:  # Check if review is not empty
            result = predict_sentiment(review)  # Predict sentiment of the review
    return render_template('index.html', result=result)  # Render the template with result

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode for easy troubleshooting
