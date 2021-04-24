from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('Model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
labels=dataframe.label
x_train,x_test,y_train,y_test=train_test_split(dataframe['text'], labels, test_size=0.2, random_state=7)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('FND.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('FND.html', prediction=pred)
    else:
        return render_template('FND.html', prediction="Aww Snap!!! Something went wrong")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)