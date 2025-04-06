import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import streamlit as st
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# Configure Gemini AI
genai.configure(api_key="AIzaSyD7yvvskWyuzMypw9AyaGQ1BF54yNjIgl4")
model = genai.GenerativeModel("gemini-2.0-flash")

# Download necessary NLTK resources
nltk.download('stopwords')

# Initialize the Treebank tokenizer
tokenizer = TreebankWordTokenizer()

# ----------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------
df = pd.read_csv('vaccination_tweets.csv')

# Data cleaning
text_df = df.drop(['id', 'user_name', 'user_location', 'user_description', 'user_created',
                   'user_followers', 'user_friends', 'user_favourites', 'user_verified',
                   'date', 'hashtags', 'source', 'retweets', 'favorites', 'is_retweet'], axis=1)

# Data processing function using TreebankWordTokenizer
stop_words = set(stopwords.words('english'))
def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = tokenizer.tokenize(text)  # Using TreebankWordTokenizer
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

# Apply data processing
text_df['text'] = text_df['text'].apply(data_processing)
text_df = text_df.drop_duplicates('text')

# Stemming function
stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

# Apply stemming to the text column
text_df['text'] = text_df['text'].apply(lambda x: stemming(x))

# Sentiment analysis functions
def polarity(text):
    return TextBlob(text).sentiment.polarity

text_df['polarity'] = text_df['text'].apply(polarity)

def sentiment(label):
    if label < 0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label > 0:
        return "Positive"

text_df['sentiment'] = text_df['polarity'].apply(sentiment)

# Feature extraction
vect = CountVectorizer(ngram_range=(1, 2)).fit(text_df['text'])
X = vect.transform(text_df['text'])
Y = text_df['sentiment']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ----------------------------------------------
# Streamlit Layout and Widgets
# ----------------------------------------------
st.set_page_config(page_title="Covid 19 Vaccination Sentiment Analysis", page_icon="📊", layout="wide")

# Title with styling
st.markdown('<h1 style="text-align: center; color: #ff6347;">Covid 19 Vaccination Tweet Sentiment Analysis</h1>', unsafe_allow_html=True)

# Sidebar options
st.sidebar.header("🔧 App Navigation")
sidebar_option = st.sidebar.radio("Choose an Option:", ['Raw Data', 'Model Selection', 'Sentiment Distribution', 'Sentiment Analysis with Gemini AI'])

# Sidebar model selection
if sidebar_option == 'Model Selection':
    model_choice = st.sidebar.radio("Choose Model:", ('Logistic Regression', 'SVC'))

# Function to plot Pie chart for Sentiment Distribution
def plot_sentiment_pie_chart():
    fig, ax = plt.subplots(figsize=(4, 4))  # Create figure and axis
    colors = ("yellowgreen", "gold", "red")
    wp = {'linewidth': 2, 'edgecolor': "black"}
    tags = text_df['sentiment'].value_counts()
    explode = (0.1, 0.1, 0.1)

    tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
              startangle=90, wedgeprops=wp, explode=explode, label='', ax=ax)  # Pass 'ax' explicitly

    ax.set_title('Distribution of sentiments', fontsize=15, fontweight='bold', color='#ff6347')

    st.pyplot(fig)  # Pass figure explicitly


# Function to plot Word Clouds for Sentiments
def plot_wordcloud(sentiment_df, sentiment_label):
    text = ' '.join([word for word in sentiment_df['text']])
    wordcloud = WordCloud(max_words=500, width=1600, height=800, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'Most Frequent Words in {sentiment_label} Tweets', fontsize=19, fontweight='bold', color='#ff6347')
    st.pyplot(fig)

# Function to display Classification Report
def display_classification_report(model_pred):
    st.write("### Classification Report:")
    st.text(classification_report(y_test, model_pred))


if sidebar_option == 'Sentiment Analysis with Gemini AI':
    # Function to configure the Gemini AI model based on user selection
    def configure_model(model_name):
        """
        Configures the generative AI model with the provided model name.
        
        Args:
        - model_name: The name of the model to be used.
        """
        genai.configure(api_key="AIzaSyD7yvvskWyuzMypw9AyaGQ1BF54yNjIgl4")  # Replace with your actual API key
        return genai.GenerativeModel(model_name)

    # Function to analyze sentiment using the selected Gemini model
    def analyze_sentiment(text, model):
        """
        Analyzes the sentiment of the given text using the selected Gemini model.
        
        Args:
        - text: The text to analyze.
        - model: The selected generative model.
        
        Returns:
        - sentiment: A string representing the sentiment ("Positive", "Negative", or "Neutral").
        - response: The raw response from the model.
        """
        try:
            prompt = f"""
            Analyze the sentiment of the following text:

            {text}

            Provide a single word answer indicating the sentiment as either "Positive", "Negative", or "Neutral".
            """

            # Generate the response
            response = model.generate_content(prompt)

            # Extract the sentiment from the response
            sentiment = response.text.strip().lower()

            # Standardize the sentiment
            if "positive" in sentiment:
                sentiment = "Positive"
            elif "negative" in sentiment:
                sentiment = "Negative"
            elif "neutral" in sentiment:
                sentiment = "Neutral"
            else:
                st.write(f"Warning: Unrecognized sentiment from model: {response.text}")  # Debugging
                return None, response

            return sentiment, response

        except Exception as e:
            st.write(f"Error during sentiment analysis: {e}")
            return None, None

    # Streamlit UI to choose model and input text
    st.title("Gemini AI Sentiment Analysis")

    # Sidebar model selection
    model_choice = st.sidebar.selectbox(
        "Choose a Gemini Model for Sentiment Analysis:",
        ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    )

    # Configure the selected model
    model = configure_model(model_choice)

    # Text input for sentiment analysis
    user_input = st.text_area("Enter the text to analyze sentiment:")

    if st.button("💬 Analyze Sentiment", key="analyze_sentiment"):
        if user_input:
            sentiment, response = analyze_sentiment(user_input, model)

            if sentiment:
                st.write(f"**Text:** '{user_input}'")
                st.write(f"**Sentiment:** {sentiment}")
                if response:
                    st.write(f"**Raw Response from Model:** {response.text}")
        else:
            st.write("Please enter text to analyze.")


# Display raw data option in sidebar
elif sidebar_option == 'Raw Data':
    st.subheader("Raw Data")
    st.write(text_df.head())

# Model training and prediction
elif sidebar_option == 'Model Selection':
    if model_choice == 'Logistic Regression':
        model = LogisticRegression()
        model.fit(x_train, y_train)
        model_pred = model.predict(x_test)
        model_acc = accuracy_score(model_pred, y_test)
        st.write(f"**Logistic Regression Test Accuracy:** {model_acc * 100:.2f}%")
        # Show word clouds after model selection
        pos_tweets = text_df[text_df.sentiment == 'Positive']
        plot_wordcloud(pos_tweets, "Positive")

        neg_tweets = text_df[text_df.sentiment == 'Negative']
        plot_wordcloud(neg_tweets, "Negative")

        neutral_tweets = text_df[text_df.sentiment == 'Neutral']
        plot_wordcloud(neutral_tweets, "Neutral")

    elif model_choice == 'SVC':
        model = LinearSVC()
        model.fit(x_train, y_train)
        model_pred = model.predict(x_test)
        model_acc = accuracy_score(model_pred, y_test)
        st.write(f"**SVC Test Accuracy:** {model_acc * 100:.2f}%")
        # Show word clouds after model selection
        pos_tweets = text_df[text_df.sentiment == 'Positive']
        plot_wordcloud(pos_tweets, "Positive")

        neg_tweets = text_df[text_df.sentiment == 'Negative']
        plot_wordcloud(neg_tweets, "Negative")

        neutral_tweets = text_df[text_df.sentiment == 'Neutral']
        plot_wordcloud(neutral_tweets, "Neutral")

# Sentiment distribution plot option (Pie chart)
elif sidebar_option == 'Sentiment Distribution':
    plot_sentiment_pie_chart()
