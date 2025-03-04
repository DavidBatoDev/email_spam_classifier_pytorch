from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Define the SpamClassifier model (must match your training definition)
class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
input_dim = 5000  # Must match the feature dimension used in training
model = SpamClassifier(input_dim)
model.load_state_dict(torch.load('spam_classifier.pth', map_location=torch.device('cpu')))
model.eval()

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the email text from the POST request
    data = request.get_json(force=True)
    email_text = data.get('email_text', '')

    # Preprocess and vectorize the email text
    processed_text = preprocess_text(email_text)
    features = vectorizer.transform([processed_text])
    features_tensor = torch.tensor(features.toarray(), dtype=torch.float32)

    # Run the model to get a prediction
    with torch.no_grad():
        output = model(features_tensor).item()

    # Convert probability to binary prediction
    prediction = 1 if output >= 0.5 else 0

    # Return the prediction and probability as JSON
    return jsonify({'prediction': prediction, 'probability': output})

if __name__ == '__main__':
    app.run(debug=True)
