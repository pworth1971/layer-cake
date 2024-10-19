# To install only the requirements of this notebook, uncomment the lines below and run this cell

# ===========================
"""
!pip install numpy==1.19.5
!pip install wget==3.2
!pip install tensorflow==1.14.0
"""

#!pip install numpy wget tensorflow tensorflow_datasets

# ===========================

#started off from: https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
#and from: https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py

#Make the necessary imports
import os
import sys
import numpy as np
import tarfile
import wget
import warnings
warnings.filterwarnings("ignore") 
from zipfile import ZipFile
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant

from tensorflow.keras.datasets import imdb

MAX_WORDS = 10000

# Load the IMDb dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=MAX_WORDS)

# The data comes preprocessed as sequences of word indices.
print(train_data[0])  # This will print an integer-encoded review


BASE_DIR = '../../layer-cake/.vector_cache/'

DATA_DIR = '../datasets/IMDB/'

GLOVE_DIR = os.path.join(BASE_DIR, 'GloVe')

TRAIN_DATA_DIR = DATA_DIR + '/train'
TEST_DATA_DIR = DATA_DIR + '/test'

print("GLOVE_DIR: ", GLOVE_DIR)
print("TRAIN_DATA_DIR: ", TRAIN_DATA_DIR)
print("TEST_DATA_DIR: ", TEST_DATA_DIR)

#Within these, I only have a pos/ and a neg/ folder containing text files 
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000 
EMBEDDING_DIM = 100 
VALIDATION_SPLIT = 0.2

#Function to load the data from the dataset into the notebook. Will be called twice - for train and test.
"""
def get_data(data_dir):
    texts = []  # list of text samples
    labels_index = {'pos':1, 'neg':0}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            if name=='pos' or name=='neg':
                label_id = labels_index[name]
                for fname in sorted(os.listdir(path)):
                        fpath = os.path.join(path, fname)
                        text = open(fpath,encoding='utf8').read()
                        texts.append(text)
                        labels.append(label_id)
    return texts, labels

train_texts, train_labels = get_data(TRAIN_DATA_DIR)
test_texts, test_labels = get_data(TEST_DATA_DIR)
"""

train_texts, train_labels = train_data, train_labels
test_texts, test_labels = test_data, test_labels

labels_index = {'pos':1, 'neg':0} 

#Just to see how the data looks like. 
print("train_texts[0]:", train_texts[0])
print("train_labels[0]", train_labels[0])

print("test_texts[24999]:", test_texts[24999])
print("test_labels[24999]:", test_labels[24999])

#
# prep IMDB data
# 

"""
import tensorflow_datasets as tfds

# Load the IMDb dataset
imdb_data = tfds.load("imdb_reviews", as_supervised=True)

# Split the data into train and test sets
train_data, test_data = imdb_data['train'], imdb_data['test']

# Extract the reviews and labels from the dataset (decode from bytes)
train_texts = [text.decode('utf-8') for text, label in tfds.as_numpy(train_data)]
test_texts = [text.decode('utf-8') for text, label in tfds.as_numpy(test_data)]

# Extract the labels
train_labels = [label for text, label in tfds.as_numpy(train_data)]
test_labels = [label for text, label in tfds.as_numpy(test_data)]

num_classes = 2
print("num_classes:", num_classes)

# Get the class names from the 20 Newsgroups dataset
class_names = ['neg', 'pos']

y_train = train_labels
y_test = test_labels

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

classification_type = 'singlelabel'
"""

from sklearn.datasets import fetch_20newsgroups

MAX_WORDS=20000
MAX_FEATURES=MAX_WORDS

TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2

# Fetch the data with return_X_y=True returns a tuple (data, labels)
train_data, train_labels = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), return_X_y=True)
test_data, test_labels = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), return_X_y=True)

# Now, train_data and test_data contain the raw text
train_texts = train_data
test_texts = test_data

# Now, train_texts and test_texts can be fed into a tokenizer.
print("train_texts:", type(train_texts), len(train_texts))
print("train_texts[0]:", type(train_texts[0]), train_texts[0])

print("test_texts:", type(test_texts), len(test_texts))
print("test_texts[0]:", type(test_texts[0]), test_texts[0])

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.utils import to_categorical

# Fetch the dataset
train_data, train_labels = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), return_X_y=True)
test_data, test_labels = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), return_X_y=True)

# Convert labels to numpy.int64
train_labels = train_labels.tolist()
test_labels = test_labels.tolist()

# Print the types and values
print("train_labels:", type(train_labels), len(train_labels))
print("train_labels[0]:", type(train_labels[0]), train_labels[0])

print("test_labels:", type(test_labels), len(test_labels))
print("test_labels[0]:", type(test_labels[0]), test_labels[0])

num_classes = 20
print("num_classes:", num_classes)

# Get the class names from the 20 Newsgroups dataset
class_names = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).target_names
print("class_names:", class_names)

# One-hot encode the labels
y_train = to_categorical(train_labels, num_classes)
y_test = to_categorical(test_labels, num_classes)

# Print the shape and type of one-hot encoded labels
print("y_train:", type(y_train), y_train.shape)
print("y_train[0]:", type(y_train[0]), y_train[0].shape)
print("y_train[0]:", y_train[0])

print("y_test:", type(y_test), y_test.shape)
print("y_test[0]:", type(y_test[0]), y_test[0].shape)
print("y_test[0]:", y_test[0])

classification_type = 'singlelabel'

#Vectorize these text samples into a 2D integer tensor using Keras Tokenizer 
#Tokenizer is fit on training data only, and that is used to tokenize both train and test data. 
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS) 
tokenizer.fit_on_texts(train_texts) 
train_sequences = tokenizer.texts_to_sequences(train_texts) #Converting text to a vector of word indexes 
test_sequences = tokenizer.texts_to_sequences(test_texts) 
word_index = tokenizer.word_index 
print('Found %s unique tokens.' % len(word_index))

print("train_sequences:", type(train_sequences), len(train_sequences))              #This is a list of lists, one list for each review
print("train_sequences[0]:", type(train_sequences[0]), len(train_sequences[0]))     #This is a list of word indexes for the first review
print("train_sequences[0]:", train_sequences[0])                                    #This will print a list of word indexes (depends on the tokenizer)

print("test_sequences:", type(test_sequences), len(test_sequences))                       #This is a list of lists, one list for each review
print("test_sequences[0]:", type(test_sequences[0]), len(test_sequences[0]))              #This is a list of word indexes for the 25000th review
print("test_sequences[0]:", test_sequences[0])                                            #This will print a list of word indexes (depends on the tokenizer)

#Converting this to sequences to be fed into neural network. Max seq. len is 1000 as set earlier
#initial padding of 0s, until vector is of size MAX_SEQUENCE_LENGTH
trainvalid_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
trainvalid_labels = to_categorical(np.asarray(train_labels))
test_labels = to_categorical(np.asarray(test_labels))

print("trainvalid_data:", type(trainvalid_data), trainvalid_data.shape)                         #Will print a 2D tensor
print("trainvalid_data[0]:", type(trainvalid_data[0]), trainvalid_data[0].shape)                #Will print a 1D tensor
print("trainvalid_data[0]:", trainvalid_data[0])                                                #Will print a 1D tensor with values as indexes

print("test_data:", type(test_data), test_data.shape)                               #Will print a 2D tensor
print("test_data[0]:", type(test_data[0]), test_data[0].shape)                      #Will print a 1D tensor
print("test_data[0]:\n", test_data[0])                                                #Will print a 1D tensor with values as indexes

# split the training data into a training set and a validation set
indices = np.arange(trainvalid_data.shape[0])
print("indices:", type(indices), indices.shape)
print("indices[0]:", type(indices[0]), indices[0].shape)
print("indices[0]:", indices[0])

np.random.shuffle(indices)

trainvalid_data = trainvalid_data[indices]
trainvalid_labels = trainvalid_labels[indices]

num_validation_samples = int(VALIDATION_SPLIT * trainvalid_data.shape[0])
print("num_validation_samples:", num_validation_samples)

x_train = trainvalid_data[:-num_validation_samples]
y_train = trainvalid_labels[:-num_validation_samples]
x_val = trainvalid_data[-num_validation_samples:]
y_val = trainvalid_labels[-num_validation_samples:]

#This is the data we will use for CNN and RNN training
print("x_train:", type(x_train), x_train.shape)
print("x_train[0]:", type(x_train[0]), x_train[0].shape)
print("y_train:", type(y_train), y_train.shape)
print("y_train[0]:", y_train[0])
print("x_val:", type(x_val), x_val.shape)
print("y_val:", type(y_val), y_val.shape)
print("y_val[0]:", y_val[0])

print('Preparing embedding matrix.')

GLOVE_MODEL = 'glove.6B.100d.txt'
print("GLOVE_MODEL: ", GLOVE_MODEL)

# first, build index mapping words in the embeddings set
# to their embedding vector
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, GLOVE_MODEL),encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors in Glove embeddings.' % len(embeddings_index))
#print(embeddings_index["google"])

# prepare embedding matrix - rows are the words from word_index, columns are the embeddings of that word from glove.
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load these pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print("Preparing of embedding matrix is done")

print("embedding_layer:", type(embedding_layer))
print("embedding_layer config:\n", embedding_layer.get_config())

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping

# Function to detect and set the best available device
def set_device():
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU (CUDA)")
        return "/device:GPU:0"
    elif tf.config.list_physical_devices('MPS'):
        print("Using Apple MPS (Metal Performance Shaders)")
        return "/device:GPU:0"  # MPS is identified as a GPU device in TensorFlow
    else:
        print("Using CPU")
        return "/device:CPU:0"
    
# Set the device
device_name = set_device()
print("Running on device:", device_name)


# Custom F1 Score Callback
class F1ScoreCallback(Callback):

    def __init__(self, validation_data, threshold=0.5):
        super().__init__()
        self.validation_data = validation_data
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_data, val_labels = self.validation_data
        
        """
        print("val_data:", type(val_data), val_data.shape)
        print("val_data[0]:", val_data[0])
        print("val_labels:", type(val_labels), val_labels.shape)
        print("val_labels[0]:", val_labels[0])
        """

        val_predictions = self.model.predict(val_data)
        #print("val_predictions:", type(val_predictions), val_predictions.shape)
        #print("val_predictions[0]:", val_predictions[0])

        # Thresholding for multi-label classification
        val_pred_classes = (val_predictions > self.threshold).astype(int)
        #print("val_pred_classes:", type(val_pred_classes), val_pred_classes.shape)
        #print("val_pred_classes[0]:", val_pred_classes[0])

        # Calculate macro and micro F1 scores
        macro_f1 = f1_score(val_labels, val_pred_classes, average='macro')
        micro_f1 = f1_score(val_labels, val_pred_classes, average='micro')
        
        # Log F1 scores
        print(f"Epoch {epoch + 1}: Macro F1 Score = {macro_f1:.4f}, Micro F1 Score = {micro_f1:.4f}")

print("test_data:", type(test_data), test_data.shape)
print("test_data[0]:", type(test_data[0]), test_data[0].shape)
print("test_data[0]:", test_data[0])
print("train_labels:", type(train_labels), len(train_labels))
print("train_labels[0]:", type(train_labels[0]), train_labels[0])
print("test_labels:", type(test_labels), test_labels.shape)
print("test_labels[0]:", test_labels[0])

EPOCHS = 50
BATCH_SIZE = 128

from sklearn.utils.class_weight import compute_class_weight

# Callbacks for learning rate refinement, early stopping, and F1 score tracking
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=5)

f1_callback = F1ScoreCallback(validation_data=(test_data, test_labels))                     # Custom F1 score callback

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("class_weights_dict:\n", class_weights_dict)

import matplotlib.pyplot as plt

def plot_history(history):

    print(f'plotting history: {history}')

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()

    acc = history.history["acc"]
    val_acc = history.history["val_acc"]

    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

print("labels_index:", labels_index)
print("class_names:", class_names) 
print("num_classes:", num_classes)

print("x_train:", type(x_train), x_train.shape)
print("x_train[0]:", type(x_train[0]), x_train[0].shape)
print("y_train:", type(y_train), y_train.shape)
print("y_train[0]:", y_train[0])

print("x_val:", type(x_val), x_val.shape)
print("x_val[0]:", type(x_val[0]), x_val[0].shape)
print("y_val:", type(y_val), y_val.shape)
print("y_val[0]:", y_val[0])

print("train_labels:", type(train_labels), len(train_labels))
print("train_labels[0]:", train_labels[0])

#print("x_test:", type(x_test), x_test.shape)
#print("x_test[0]:", type(x_test[0]), x_test[0].shape)
print("y_test:", type(y_test), len(y_test))
print("y_test[0]:", y_test[0])
print("test_labels:", type(test_labels), len(test_labels))
print("test_labels[0]:", test_labels[0])

print('Define a 1D CNN model.')

from sklearn.metrics import f1_score, classification_report

with tf.device(device_name):

    cnnmodel = Sequential()
    cnnmodel.add(embedding_layer)
    cnnmodel.add(Conv1D(128, 5, activation='relu'))
    cnnmodel.add(MaxPooling1D(5))
    cnnmodel.add(Conv1D(128, 5, activation='relu'))
    cnnmodel.add(MaxPooling1D(5))
    cnnmodel.add(Conv1D(128, 5, activation='relu'))
    cnnmodel.add(GlobalMaxPooling1D())
    cnnmodel.add(Dense(128, activation='relu'))
    cnnmodel.add(Dense(num_classes, activation='softmax'))

    cnnmodel.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
    
    """
    #Train the model. Tune to validation set. 
    cnnmodel.fit(x_train, y_train,
            batch_size=128,
            epochs=10, validation_data=(x_val, y_val))
    """

    # Then pass class_weights_dict to the fit function
    history = cnnmodel.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=[reduce_lr, early_stop, f1_callback],
        class_weight=class_weights_dict
        )
    
print("history:", type(history), history.history.keys())


plot_history(history)

print("y_test:", type(y_test), y_test.shape)
print("y_test[0]:", y_test[0])

#Evaluate on test set:
score, acc = cnnmodel.evaluate(test_data, test_labels)
print("score:", score)
print('Test accuracy with CNN:', acc)


test_preds = cnnmodel.predict(test_data)
        
# Thresholding for multi-label classification
test_pred_classes = (test_preds > .5).astype(int)

# Classification report with class labels and four decimal places
report = classification_report(test_labels, test_pred_classes, target_names=class_names, digits=4)
print("\nClassification Report:\n", report)

# Calculate macro and micro F1 scores
macro_f1 = f1_score(y_test, test_pred_classes, average='macro')
micro_f1 = f1_score(y_test, test_pred_classes, average='micro')

# Log F1 scores
print(f"\n\tMacro F1 Score = {macro_f1:.4f}, Micro F1 Score = {micro_f1:.4f}")

print("x_train:", type(x_train), x_train.shape)
print("x_train[0]:", type(x_train[0]), x_train[0].shape)
#print("x_train[0]:\n", x_train[0])

print("y_train:", type(y_train), y_train.shape)
print("y_train[0]:", y_train[0])

print("x_val:", type(x_val), x_val.shape)
print("x_val[0]:", type(x_val[0]), x_val[0].shape)
#print("x_val[0]:\n", x_val[0])

print("class_weights_dict:\n", class_weights_dict)

print(f"Using device: {device_name}")

print("Defining and training a CNN model, training embedding layer on the fly instead of using pre-trained embeddings")

# Define the CNN model
cnnmodel = Sequential()

# Force the Embedding layer to run on the CPU
with tf.device('/CPU:0'):
    cnnmodel.add(Embedding(MAX_NUM_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH))

# Rest of the model can run on the GPU
with tf.device(device_name):
    cnnmodel.add(Conv1D(128, 5, activation='relu'))
    cnnmodel.add(MaxPooling1D(5))
    cnnmodel.add(Conv1D(128, 5, activation='relu'))
    cnnmodel.add(MaxPooling1D(5))
    cnnmodel.add(Conv1D(128, 5, activation='relu'))
    cnnmodel.add(GlobalMaxPooling1D())
    cnnmodel.add(Dense(128, activation='relu'))
    cnnmodel.add(Dense(num_classes, activation='softmax'))

    cnnmodel.compile(loss='categorical_crossentropy',
                     optimizer='rmsprop',
                     metrics=['acc'])

    #Train the model. Tune to validation set. 
    history2 = cnnmodel.fit(
        x_train, 
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS, 
        validation_data=(x_val, y_val),
        callbacks=[reduce_lr, early_stop, f1_callback],
        class_weight=class_weights_dict
        )
    
print("history2:", type(history2), history2.history.keys())
plot_history(history2)

print("test_data:", type(test_data), test_data.shape)
print("test_data[0]:", type(test_data[0]), test_data[0].shape)
print("test_data[0]:", test_data[0])

print("test_labels:", type(test_labels), len(test_labels))
print("test_labels[0]:", test_labels[0])

#Evaluate on test set:
score, acc = cnnmodel.evaluate(test_data, test_labels)
print("score:", score)
print('Test accuracy with CNN:', acc)


test_preds = cnnmodel.predict(test_data)
        
# Thresholding for multi-label classification
test_pred_classes = (test_preds > .5).astype(int)

# Classification report with class labels and four decimal places
report = classification_report(test_labels, test_pred_classes, target_names=class_names, digits=4)
print("\nClassification Report:\n", report)

# Calculate macro and micro F1 scores
macro_f1 = f1_score(y_test, test_pred_classes, average='macro')
micro_f1 = f1_score(y_test, test_pred_classes, average='micro')

# Log F1 scores
print(f"\n\tMacro F1 Score = {macro_f1:.4f}, Micro F1 Score = {micro_f1:.4f}")


physical_devices = tf.config.list_physical_devices('GPU')
print("physical_devices:", physical_devices)

if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(f'Error: {e}')

tf.debugging.set_log_device_placement(True)

print("Defining and training an LSTM model, training embedding layer on the fly")

LSTM_BATCH_SIZE = 32

# Define the RNN model
rnnmodel = Sequential()

# Force the Embedding layer to run on the CPU
with tf.device('/CPU:0'):
        rnnmodel.add(Embedding(MAX_NUM_WORDS, 128))
        
with tf.device(device_name):
        #rnnmodel.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        rnnmodel.add(LSTM(128, dropout=0.2))
        rnnmodel.add(Dense(num_classes, activation='sigmoid'))

rnnmodel.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )

print('Training the RNN')

rnnmodel.fit(
  x_train, 
  y_train,
  batch_size=LSTM_BATCH_SIZE,
  epochs=EPOCHS,
  validation_data=(x_val, y_val),
  callbacks=[reduce_lr, early_stop, f1_callback],
  class_weight=class_weights_dict
)

#Evaluate on test set:
score, acc = rnnmodel.evaluate(test_data, test_labels)
print("score:", score)
print('Test accuracy with CNN:', acc)

test_preds = rnnmodel.predict(test_data)
        
# Thresholding for multi-label classification
test_pred_classes = (test_preds > .5).astype(int)

# Classification report with class labels and four decimal places
report = classification_report(test_labels, test_pred_classes, target_names=class_names, digits=4)
print("\nClassification Report:\n", report)

# Calculate macro and micro F1 scores
macro_f1 = f1_score(y_test, test_pred_classes, average='macro')
micro_f1 = f1_score(y_test, test_pred_classes, average='micro')

# Log F1 scores
print(f"\n\tMacro F1 Score = {macro_f1:.4f}, Micro F1 Score = {micro_f1:.4f}")

print("Defining and training an LSTM model, using pre-trained embedding layer")

rnnmodel2 = Sequential()
rnnmodel2.add(embedding_layer)
#rnnmodel2.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
rnnmodel2.add(LSTM(128, dropout=0.2))
rnnmodel2.add(Dense(num_classes, activation='sigmoid'))

rnnmodel2.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
  )

print('Training the RNN')

history3 = rnnmodel2.fit(
  x_train, 
  y_train,
  batch_size=LSTM_BATCH_SIZE,
  epochs=EPOCHS,
  validation_data=(x_val, y_val),
  callbacks=[reduce_lr, early_stop, f1_callback],
  class_weight=class_weights_dict
  )

print("history:", type(history3), history3.history.keys())
plot_history(history3)

#Evaluate on test set:
score, acc = rnnmodel2.evaluate(test_data, test_labels, batch_size=LSTM_BATCH_SIZE)
print("score:", score)
print('Test accuracy with CNN:', acc)

test_preds = rnnmodel2.predict(test_data)
        
# Thresholding for multi-label classification
test_pred_classes = (test_preds > .5).astype(int)

# Classification report with class labels and four decimal places
report = classification_report(test_labels, test_pred_classes, target_names=class_names, digits=4)
print("\nClassification Report:\n", report)

# Calculate macro and micro F1 scores
macro_f1 = f1_score(y_test, test_pred_classes, average='macro')
micro_f1 = f1_score(y_test, test_pred_classes, average='micro')

# Log F1 scores
print(f"\n\tMacro F1 Score = {macro_f1:.4f}, Micro F1 Score = {micro_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network for text classification.')
    parser.add_argument('--MX_WORDS', type=int, required=True, help='Maximum number of words in the vocabulary')
    parser.add_argument('--MAX_SEQUENCE_LENGTH', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--dataset', type=str, choices=['IMDB', '20newsgroups'], required=True, help='Dataset to use')
    parser.add_argument('--network_type', type=str, choices=['CNN', 'CNN with Pretrained Embeddings', 'LSTM', 'LSTM with Embeddings'], required=True, help='Type of network to use')
    parser.add_argument('--pretrained_embeddings', type=str, choices=['GloVe', 'fastText', 'Word2Vec'], required=False, help='Pretrained embeddings to use (if applicable)')
    
    args = parser.parse_args()
    
    main(args.MX_WORDS, args.MAX_SEQUENCE_LENGTH, args.dataset, args.network_type, args.pretrained_embeddings)
