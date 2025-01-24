import mlflow.client
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import re
import io
import nltk
from wordcloud import WordCloud
import logging
import mlflow
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
nltk.download('wordnet')
nltk.download('stopwords')


app = Flask(__name__)
CORS(app)


# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler('errors.log')
fileHandler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(streamHandler)
logger.addHandler(fileHandler)


def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations in a comment"""
    try:
        # convert to lowercase
        comment = comment.lower()

        # remove leading and trailing whitespace
        comment = comment.strip()

        # remove newline chars
        comment = re.sub(r'\n', ' ', comment)

        # remove non alpha-numeric chars except punctuation
        comment = re.sub(r'[^a-zA-Z0-9\s,.!?]', '', comment)

        # remove some stopwords
        stopwords = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stopwords])

        # apply lemmatization
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment    
    except Exception as e:
        logger.error('Error in preprocessing comment %s', e)
        return comment
    

def load_model_vectorizer(model_name: str, model_version: int, vectorizer_path: str):
    """Load model from model registry and vectorizer from given file"""
    try:
        # set mlflow tracking uri
        mlflow.set_tracking_uri('http://ec2-13-61-2-37.eu-north-1.compute.amazonaws.com:5000')

        # model
        client = mlflow.client.MlflowClient()
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # vectorizer
        with open(vectorizer_path, 'rb') as file:
            vectorizer = joblib.load(file)

        logger.debug('model and vectorizer loaded successfully')
        return model, vectorizer
    except Exception as e:
        logger.error(f'Error while loading model and vectorizer: {e}')
        raise


# model and vectorizer
model, vectorizer = load_model_vectorizer('lgbm_model', 1, 'tfidf_vectorizer.pkl')


@app.route('/')
def home():
    return "Welcome"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({'error': "No comments provided"})
    
    try:
        # preprocess comment
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # vectorizer transformation
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # predictions
        predictions = model.predict(transformed_comments).tolist()
        predictions = [str(prediction) for prediction in predictions]

        # response
        response = [{'comment': comment, 'sentiment':sentiment} for comment, sentiment in zip(comments, predictions)]
        return jsonify(response)
    except Exception as e:
        logger.error('Error in predict funct: %s', e)
        return jsonify({'error': f'Prediction failed {str(e)}'}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    comments = [item['text'] for item in comments_data]
    timestamps = [item['timestamp'] for item in comments_data]

    if not comments:
        return jsonify({'error': "No comments provided"})
    
    try:
        # preprocess comment
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # vectorizer transformation
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # predictions
        predictions = model.predict(transformed_comments).tolist()
        predictions = [str(prediction) for prediction in predictions]

        # response
        response = [{'comment': comment, 'sentiment':sentiment, 'timestamp':timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
        return jsonify(response)
    except Exception as e:
        logger.error('Error in predict_with_timestamp funct: %s', e)
        return jsonify({'error': f'Prediction with timestamp failed {str(e)}'}), 500
    

@app.route('/generate_pie_chart', methods=['POST'])
def generate_pie_chart():
    try:
        data = request.json
        sentiments_count = data.get('sentiment_counts')

        if not sentiments_count:
            return jsonify({'error':'No sentiment counts provided'}), 400
        
        # prepare data for piecharts
        positive_count = int(sentiments_count.get('1', 0))
        neutral_count = int(sentiments_count.get('0', 0))
        negative_count = int(sentiments_count.get('-1', 0))
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [positive_count, neutral_count, negative_count]

        if sum(sizes) == 0:
            raise ValueError('Sentiments sum count is zero')
        
        # generate pie chart
        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.axis('equal')

        # save the chart to the bytesIO object
        io_img_obj = io.BytesIO()
        plt.savefig(io_img_obj, format='PNG', transparent=True)
        io_img_obj.seek(0)
        plt.close()

        # return image
        return send_file(io_img_obj, mimetype='image/png')    
    except Exception as e:
        logger.error(f"Error in generating pie chart: {e}")
        return jsonify({"error": f"Pie chart generation failed: {str(e)}"}), 500
    

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.json
        comments = data.get('comments')

        if not comments:
            return jsonify({'error': "No comments provided"})
        
        # preprocess comments
        processed_comments = [preprocess_comment(comment) for comment in comments]
        text = " ".join(processed_comments)

        # generate wordcloud
        stopwords = stopwords.word('english')
        wordcloud = WordCloud(width=800, height=400, colormap='Blues', stopwords=stopwords, collocations=False, max_words=600)
        wordcloud.generate(text)

        # iostream
        img_io_obj = io.BytesIO()
        wordcloud.to_image().save(img_io_obj, format='PNG')
        img_io_obj.seek(0)

        return send_file(img_io_obj, mimetype='image/png')
    except Exception as e:
        logger.error(f'Error while generation wordcloud: {e}')
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500
    

@app.route('/generate_trend_graph', method=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:            
            return jsonify({'error': "No sentiment data provided"}), 400
        
        # convert sentiment data to dataframe
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sentiment'] = df['sentiment'].astype(int)

        # set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # sentiment labels
        sentiment_labels = {-1:'Negative', 0:'Neutral', 1:'Positive'}

        # resample data over monthly intervels and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)

        # calculate percentage
        monthly_perc = (monthly_counts.T/monthly_totals).T * 100
        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_perc.columns:
                monthly_perc[sentiment_value] = 0

        monthly_perc = monthly_perc[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_perc.index,
                monthly_perc[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error in generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500
    

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)