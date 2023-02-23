# Next Word Predictor using TensorFlow
This repository contains a Python implementation of a next word predictor using TensorFlow. The model is trained on a large corpus of text data, and it uses a neural network to predict the most likely next word given a sequence of previous words. This type of model can be used in a variety of natural language processing applications, such as speech recognition, machine translation, and text completion.

**Getting Started**
To use this next word predictor, follow these steps:

Clone the repository: git clone https://github.com/Atomic-man007/next_word_prediction.git

Install the required packages: pip install -r requirements.txt
Navigate to the project directory and the notebook: next_word_pred.ipynb
Start the web application: python app.py
Once the web application is running, you can access it by visiting http://localhost:5000 in your web browser. Enter a sequence of words into the input box, and the model will predict the most likely next word.

**Files**
The following files are included in this repository:

app.py: A Flask web application that uses the trained model to predict the next word given a sequence of previous words.
download_data.py: A Python script that downloads the training data.
requirements.txt: A file listing the required Python packages.

**Training Data**
The model is trained on a corpus of text data which is "Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyle".

**Model Architecture**
The next word predictor uses a recurrent neural network (RNN) with a long short-term memory (LSTM) cell. The model takes a sequence of previous words as input, and it predicts the probability distribution over the vocabulary for the next word. The output of the model is a softmax distribution over the vocabulary, and the most likely next word is the one with the highest probability.

The model is trained using a cross-entropy loss function and the Adam optimizer. The training data is split into training and validation sets, and the model's performance is evaluated on the validation set after each epoch. The training stops when the validation loss stops decreasing, or when a maximum number of epochs is reached.
