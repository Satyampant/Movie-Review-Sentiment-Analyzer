from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import datetime
import os

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
# model_path = os.getcwd()+r'\sentimentanalysis\models\model'
model_path = os.getcwd() + r'\models'
classifier = joblib.load(model_path + r'\classifier.pkl')


def predictfunc(review):
    prediction = classifier.predict(review)
    if prediction[0] == 1:
        sentiment = 'Positive'
        prob = classifier.predict_proba(review)[:, 1]
        img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.gif')
        emoji = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')

    else:
        sentiment = 'Negative'
        prob = classifier.predict_proba(review)[:, 0]
        img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.gif')
        emoji = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
    return prob, sentiment, img_filename, emoji


@app.route('/', methods = ['POST', "GET"])
def index():
    return render_template('index.html')


@app.route('/sentiment_prediction', methods = ['POST', "GET"])
def predict():
    if request.method == 'POST':
        result = request.form
        content = request.form['text']
        review = pd.Series(content)
        prob, sentiment, img_filename, emoji = predictfunc(review)
    return render_template("index.html", text=content, sentiment=sentiment, probability=prob, image=img_filename, emoji=emoji)


if __name__ == '__main__':
    #app.run(debug = True,port=5000)
    app.run(host='0.0.0.0')
