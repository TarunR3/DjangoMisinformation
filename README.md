6. Open your web browser and go to `https://fakenewstf.web.app/` to start using the app.

## Usage

- **Detecting Fake News**:
1. Navigate to the home page of the application.
2. Enter the URL of the news article you want to check or paste the text directly into the provided field.
3. Click on the "Classify" button to submit the information.
4. View the classification result displayed on the screen.

## How It Works

The application processes the input text using NLTK to tokenize the text and filter out stopwords. This processed text is then fed into a pre-trained LSTM model built with TensorFlow, which classifies the text as 'fake' or 'real' based on its learning from a vast dataset of labeled news articles.