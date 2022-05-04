from flask import Flask, render_template, url_for, request, jsonify
import pickle
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
gender = {0: "male", 1: "female", 2: "brand"}

@app.route('/')
def index():
  return render_template('menu.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        input = request.get_data()
        input = input.decode('ascii')
        out = pred(input)
        return out


def pred(input):
  with open('./models/model_rfc2', 'rb') as f:
    model = pickle.load(f)
  max_words = 10000
  max_len = 200

  tokenizer = Tokenizer(num_words = max_words)
  tokenizer.fit_on_texts(input)
  sequences = tokenizer.texts_to_sequences(input)
  result = pad_sequences(sequences, maxlen = max_len)
  y_pred = model.predict(result)
  y_pred = y_pred.tolist()
  common = max(y_pred,  key = y_pred.count)
  return gender[common]

if __name__ == "__main__":
  app.run(debug = True)