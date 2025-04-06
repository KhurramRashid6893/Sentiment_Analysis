## 📊 Covid-19 Vaccination Tweet Sentiment Analysis (Flask App)

A web-based sentiment analysis tool for Covid-19 vaccination-related tweets. Built with **Flask**, uses **NLP**, **Machine Learning**, and **Gemini AI** for real-time sentiment prediction.

> 🔬 Dataset Source: [Kaggle - Vaccination Tweets CSV](https://www.kaggle.com/datasets/abdeltawabali/vaccination-tweets-csv)

---

## 🚀 Features

### 🏠 Home Page
- Central hub with navigation to different features.
- Clean UI with styled buttons and layout.

### 🧾 Raw Data Viewer
- Displays cleaned tweet text.
- User can select how many rows to view (5–100).
- Original Kaggle dataset link included.

### 📊 Sentiment Distribution
- Visualizes sentiment distribution (Positive, Negative, Neutral) with a pie chart.

### ⚙️ Model Selection
- Choose between **Logistic Regression** and **Support Vector Machine (SVC)**.
- View model accuracy.
- Word clouds for each sentiment category.
- Classification report for model evaluation.

### 🤖 Gemini AI Sentiment Analysis
- Use Google's Gemini AI (via API) for sentiment analysis.
- Enter custom text and get sentiment in real-time (Positive / Neutral / Negative).
- View raw response from the AI.

### 🧾 Footer
- Custom footer with author name and current date, present on every page.

---

## 📁 Project Structure

```
sentiment-app/
│
├── app.py
├── vaccination_tweets.csv
├── requirements.txt
├── Procfile
├── templates/
│   ├── index.html
│   ├── raw_data.html
│   ├── sentiment_distribution.html
│   ├── model_selection.html
│   ├── gemini_sentiment.html
│   └── footer.html
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sentiment-app.git
cd sentiment-app
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

```python
import nltk
nltk.download('stopwords')
```

### 5. Run the app

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## 🌐 Deployment on Render

### Required Files:
- `Procfile`
- `requirements.txt`

### Steps:
1. Push code to GitHub
2. Go to [Render](https://render.com/)
3. Create a new Web Service → Connect GitHub
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `python app.py`
6. Deploy 🚀

---

## 🤝 Acknowledgements

- [Kaggle - Vaccination Tweets Dataset](https://www.kaggle.com/datasets/abdeltawabali/vaccination-tweets-csv)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- Libraries: Flask, NLTK, TextBlob, Scikit-learn, WordCloud, Matplotlib

---

## 👨‍💻 Author

**Khurram Rashid**  
Feel free to connect or contribute!

