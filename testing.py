import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the trained model from the saved file
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the vectorizer used during training
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to clean and preprocess text (same as used during model training)
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    # Remove stopwords (common words like "the", "and" that do not add much meaning)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))  
    return text

# Load the testing dataset
test_df = pd.read_csv("test.csv")  

# Take a random sample of 30,000 rows for evaluation
test_df = test_df.sample(30000, random_state=42)

# Ensure that the required columns exist in the dataset
if 'Text' not in test_df.columns or 'Sentiment' not in test_df.columns:
    raise ValueError("The CSV file must contain 'Text' and 'Sentiment' columns.")

# Convert 'Sentiment' column to numeric type (if it is not already)
test_df['Sentiment'] = test_df['Sentiment'].astype(int)

# Apply text preprocessing to the test dataset
test_df['Cleaned_Text'] = test_df['Text'].apply(clean_text)

# Convert cleaned text into numerical features using the pre-trained TF-IDF vectorizer
X_test = vectorizer.transform(test_df['Cleaned_Text'])

# Predict sentiment labels using the trained model
predictions = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(test_df['Sentiment'], predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")  # Print accuracy as percentage

# Print detailed classification report (precision, recall, f1-score)
print("\nClassification Report:\n")
print(classification_report(test_df['Sentiment'], predictions, target_names=["Negative", "Positive"]))

# Visualization: Pie chart showing model accuracy
labels = ['Accurate', 'Non-Accurate']
sizes = [accuracy * 100, 100 - accuracy * 100]  # Divide data into correct and incorrect predictions
colors = ['green', 'red']  # Green for correct predictions, red for incorrect
explode = (0.1, 0)  # Slightly separate the "Accurate" slice for emphasis

# Plotting the pie chart
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Ensures the pie chart is circular
plt.title("Model Accuracy Visualization")
plt.show()  # Display the pie chart
