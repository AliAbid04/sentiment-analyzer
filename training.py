import pandas as pd
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from nltk.corpus import stopwords

# Downloading necessary NLTK data (stopwords and lemmatizer)
nltk.download('stopwords')
nltk.download('wordnet')

# Reading the dataset
df = pd.read_csv("train.csv")

# Dropping rows where 'Sentiment' or 'Text' is missing
df = df.dropna(subset=['Sentiment', 'Text'])

# Ensuring the 'Sentiment' column is of integer type
df['Sentiment'] = df['Sentiment'].astype(int)

# Sampling 300,000 rows for training to manage large dataset size
df = df.sample(300000, random_state=42)

# Function for text preprocessing
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    # Remove stopwords (common words that do not add much meaning, e.g., "the", "and")
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

# Applying the text cleaning function to the dataset
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Converting text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
vectorizer = TfidfVectorizer(max_features=10000)  # Limiting to 10,000 features for efficiency
X = vectorizer.fit_transform(df['Cleaned_Text'])  # Transform the cleaned text into a matrix
y = df['Sentiment']  # Defining target variable (Sentiment)

# Splitting data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating evaluation metrics
mse = mean_squared_error(y_test, y_pred)  # Measures average squared difference between actual & predicted values
r2 = r2_score(y_test, y_pred)  # Measures how well predictions fit the actual data

# Displaying model performance
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Saving the trained model and vectorizer for future use
with open("sentiment_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")
