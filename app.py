from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow

app = Flask(__name__)

# Load the trained model
model = tensorflow.keras.models.load_model('next_word_pred_150.h5', compile=False)

tokenizer = Tokenizer()

# Load the word index mapping
with open('word_index.json', 'r') as f:
    word_index = json.load(f)

# Define the maximum length of input sequence
max_length = 20
vocab_array = np.array(list(tokenizer.word_index.keys()))

def make_prediction(text, n_words):
    for i in range(n_words):
        text_tokenize = tokenizer.texts_to_sequences([text])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=14)
        prediction = np.squeeze(np.argmax(model.predict(text_padded), axis=-1))
        prediction = str(vocab_array[prediction - 1])
        print(vocab_array[np.argsort(model.predict(text_padded)) - 1].ravel()[:-3])
        text += " " + prediction
    return text

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input sequence from the request
    input_sequence = request.json['input_sequence']

    # Convert the input sequence to a sequence of integers
    input_sequence = [word_index.get(word, 0) for word in input_sequence.split()]

    predicted_word = make_prediction(input_sequence, 1)

    # Return the predicted word
    return jsonify({'predicted_word': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)
