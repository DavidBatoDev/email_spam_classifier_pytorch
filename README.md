# Email Spam Classifier API

This repository contains a Flask-based API that serves a pretrained email spam classifier that I build using simple feedforward neural network MLP. The classifier uses a PyTorch model and a TF-IDF vectorizer to determine whether a given email is spam (1) or ham (0).

![image](https://github.com/user-attachments/assets/0caf7617-a2bb-4c4e-935e-15ed6f20e452)

This project covers data preprocessing, text vectorization, model training, and evaluation using a simple feedforward neural network (MLP). The [Email Spam Dataset](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset/data) from Kaggle for training and testing

## Overview

The API provides a single endpoint `/predict` that accepts POST requests with JSON payloads containing an email text. The text is preprocessed, vectorized, and passed to the model to predict if the email is spam. The response returns a binary prediction (1 for spam, 0 for ham) along with the probability score.

> **Important:** If you are using a notebook to run your setup cells (for example, installing dependencies or downloading NLTK data), ensure that you run **every cell** in the notebook before running the `app.py` file. This guarantees that all required libraries, configurations, and data (like NLTK packages) are properly loaded.

## Prerequisites

- **Python 3.7+**
- **PyTorch** (Installed via `pip install torch`)
- **Flask** (Installed via `pip install flask`)
- **NLTK** (Installed via `pip install nltk`)
- **Scikit-Learn** (Installed via `pip install scikit-learn`)

> **Note:** Ensure you have the pretrained model file (`spam_classifier.pth`) and the TF-IDF vectorizer (`tfid_vectorizer.pkl`) in the root directory of the project.

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/DavidBatoDev/email_spam_classifier.git
   cd email_spam_classifier
   ```
   
2. **Install Dependencies:**
Install the required packages via the provided `requirements.txt`
  ```bash
    pip install -r requirements.txt
  ```
3. **Download NLTK Data:**

The app downloads required NLTK packages (punkt, stopwords, wordnet) on startup. If running in a notebook, make sure you run the cell that downloads these packages before starting the Flask app:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

## Running the Application
To start the Flask app, run the following command from the project directory:
  ```bash
  python app.py
  ```
You should see output similar to:
  ```csharp
   * Serving Flask app 'app'
   * Debug mode: on
   * Running on http://127.0.0.1:5000
  ```

## API Endpoint
### Post `/predict`
* Description: Predicts whether an email is spam.

* URL: `http://127.0.0.1:5000/predict`
* Method: `POST`
* Headers: `Content-Type: application/json`
* Body Example:
For testing spam:
  ```json
  {
    "email_text": "Congratulations! You have been selected for a free prize. Click here to claim your reward."
  }
  ```
For testing ham:
  ```json
  {
    "email_text": "Hi team, please find attached the minutes from yesterday's meeting."
  }
  ```
* Response Example:
  ```json
  {
    "prediction": 1,           // 1 indicates spam, 0 indicates ham
    "probability": 0.87        // Probability score from the model
  }
  ```
## Testing the API
### Using Insomnia or Postman
1. **Create a New Request:**

- Method: POST
- URL: `http://127.0.0.1:5000/predict`
  
2. **Set the Request Body (JSON):**
Insert the appropriate JSON payload for testing spam or ham.

3. **Send the Request:**
Verify that the response correctly reflects whether the email is spam or ham.

## Security Note
Security Note
You may see a warning regarding the use of torch.load with the default settings. For production deployments, consider using:
```python
model.load_state_dict(torch.load('./spam_classifier.pth', map_location=torch.device('cpu'), weights_only=True))
```

## Contact
For any issues or questions, please open an issue in the GitHub repository or contact batobatodavid20@gmail.com
