import pandas as pd
import numpy as np
import re
import nltk
from flask import Flask, render_template, request
from textblob import TextBlob
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import warnings
import os
import io
import base64

warnings.filterwarnings('ignore')

# Setup Flask
app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords')
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Configure Gemini AI
genai.configure(api_key="AIzaSyD7yvvskWyuzMypw9AyaGQ1BF54yNjIgl4")

# Load and process data
df = pd.read_csv('vaccination_tweets.csv')
text_df = df.drop(['id', 'user_name', 'user_location', 'user_description', 'user_created',
                   'user_followers', 'user_friends', 'user_favourites', 'user_verified',
                   'date', 'hashtags', 'source', 'retweets', 'favorites', 'is_retweet'], axis=1)

def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = tokenizer.tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

text_df['text'] = text_df['text'].apply(data_processing)
text_df = text_df.drop_duplicates('text')

def stemming(data):
    return " ".join([stemmer.stem(word) for word in data.split()])

text_df['text'] = text_df['text'].apply(stemming)

def polarity(text):
    return TextBlob(text).sentiment.polarity

def sentiment(label):
    if label < 0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    else:
        return "Positive"

text_df['polarity'] = text_df['text'].apply(polarity)
text_df['sentiment'] = text_df['polarity'].apply(sentiment)

vect = CountVectorizer(ngram_range=(1, 2)).fit(text_df['text'])
X = vect.transform(text_df['text'])
Y = text_df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ML models
log_model = LogisticRegression()
log_model.fit(x_train, y_train)
log_pred = log_model.predict(x_test)
log_acc = accuracy_score(log_pred, y_test)

svc_model = LinearSVC()
svc_model.fit(x_train, y_train)
svc_pred = svc_model.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)

# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route("/raw-data", methods=["GET", "POST"])
def raw_data():
    num_rows = 5
    if request.method == "POST":
        num_rows = int(request.form["num_rows"])

    table_html = text_df.head(num_rows).to_html(classes="table table-striped table-bordered", index=False)
    return render_template("raw_data.html", table=table_html, selected=str(num_rows))



@app.route('/sentiment-distribution')
def sentiment_distribution():
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = ("yellowgreen", "gold", "red")
    wp = {'linewidth': 2, 'edgecolor': "black"}
    tags = text_df['sentiment'].value_counts()
    explode = (0.1, 0.1, 0.1)

    tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
              startangle=90, wedgeprops=wp, explode=explode, label='', ax=ax)
    ax.set_title('Distribution of sentiments', fontsize=15, fontweight='bold', color='#ff6347')
    plt.tight_layout()
    fig_path = os.path.join('static', 'sentiment_pie_chart.png')
    plt.savefig(fig_path)
    return render_template('sentiment_distribution.html', image_path=fig_path)

@app.route("/model-selection", methods=["GET", "POST"])
def model_selection():
    selected_model = "Logistic Regression"
    accuracy = None
    classification = None
    wordclouds = {}

    if request.method == "POST":
        selected_model = request.form["model"]
        if selected_model == "Logistic Regression":
            model = LogisticRegression()
        else:
            model = LinearSVC()
        
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
        classification = classification_report(y_test, predictions)

        # WordCloud generation for each sentiment
        sentiments = ["Positive", "Negative", "Neutral"]
        for sentiment in sentiments:
            words = " ".join(text_df[text_df["sentiment"] == sentiment]["text"])
            wc = WordCloud(width=800, height=400, background_color="white").generate(words)
            img = io.BytesIO()
            wc.to_image().save(img, format="PNG")
            img.seek(0)
            wordclouds[sentiment] = base64.b64encode(img.read()).decode("utf-8")

    return render_template("model_selection.html", 
                           selected_model=selected_model,
                           accuracy=accuracy, 
                           classification_report=classification,
                           wordclouds=wordclouds)

@app.route('/gemini-sentiment', methods=['GET', 'POST'])
def gemini_sentiment():
    sentiment = None
    user_input = ''
    raw_response = ''

    if request.method == 'POST':
        user_input = request.form['text']
        if user_input:
            prompt = f"""
            Analyze the sentiment of the following text:

            {user_input}

            Provide a single word answer indicating the sentiment as either "Positive", "Negative", or "Neutral".
            """
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            raw_response = response.text.strip()

            if "positive" in raw_response.lower():
                sentiment = "Positive"
            elif "negative" in raw_response.lower():
                sentiment = "Negative"
            elif "neutral" in raw_response.lower():
                sentiment = "Neutral"
            else:
                sentiment = "Unrecognized"

    return render_template('gemini_sentiment.html', sentiment=sentiment, user_input=user_input, raw_response=raw_response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
