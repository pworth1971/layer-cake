
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Attention, Bidirectional, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from scipy.sparse import issparse


MAX_WORDS = 20000
MAX_FEATURES=MAX_WORDS
MAX_SEQUENCE_LENGTH = 1000

TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2




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

    

def load_reuters_data(vtype="count", max_features=MAX_WORDS, debug=False):

    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=max_features)

    if (vtype == 'count'):
        vectorizer = CountVectorizer(max_features=max_features, lowercase=False, stop_words='english')
    elif (vtype == 'tfidf'):
        vectorizer = TfidfVectorizer(max_features=max_features, lowercase=False, stop_words='english')

    print("vectorizer:", vectorizer)

    # Get the word index from the Reuters dataset
    word_index = reuters.get_word_index()
    reverse_word_index = {value: key for (key, value) in word_index.items()}

    # Convert the list of word indices back into readable text (documents)
    train_data = [' '.join([reverse_word_index.get(i - 3, '?') for i in seq]) for seq in train_data]
    test_data = [' '.join([reverse_word_index.get(i - 3, '?') for i in seq]) for seq in test_data]
        
    raw_train = train_data
    raw_test = test_data

    if (debug):
        print("raw_train:", type(raw_train), len(raw_train))
        print("raw_train[0]:", type(raw_train[0]), len(raw_train[0]))
        print("raw_train[0]:", raw_train)
          
        print("raw_test:", type(raw_test), len(raw_test))
        print("raw_test[0]:", type(raw_test[0]), len(raw_test[0]))
        print("raw_test[0]:", raw_test[0])

    # Use vectorizer for feature extraction
    x_train_vectorized = vectorizer.fit_transform(train_data).toarray()
    x_test_vectorized = vectorizer.transform(test_data).toarray()

    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    
    return raw_train, x_train_vectorized, y_train, raw_test, x_test_vectorized, y_test, train_labels, test_labels, vectorizer


# Load and preprocess 20 Newsgroups data
def load_20newsgroups_data(vtype='count', max_features=MAX_WORDS, debug=False):

    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # Get the class names from the 20 Newsgroups dataset
    class_names = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).target_names
    
    raw_train = newsgroups_train.data
    raw_test = newsgroups_test.data
    
    if (debug):
        print("raw_train:", type(raw_train), len(raw_train))
        print("raw_train[0]:", type(raw_train[0]), len(raw_train[0]))
        print("raw_train[0]:", raw_train[0])
        
        print("raw_test:", type(raw_test), len(raw_test))
        print("raw_test[0]:", type(raw_test[0]), len(raw_test[0]))
        print("raw_test[0]:", raw_test[0])

    if (vtype == 'count'):
        vectorizer = CountVectorizer(max_features=max_features, lowercase=False, stop_words='english')
    elif (vtype == 'tfidf'):
        vectorizer = TfidfVectorizer(max_features=max_features, lowercase=False, stop_words='english')

    print("vectorizer:", vectorizer)

    x_train_vectorized = vectorizer.fit_transform(newsgroups_train.data).toarray()
    x_test_vectorized = vectorizer.transform(newsgroups_test.data).toarray()

    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    return raw_train, x_train_vectorized, y_train, raw_test, x_test_vectorized, y_test, newsgroups_train.target, newsgroups_test.target, class_names, vectorizer




def load_20newsgroups_data_nn(max_words=MAX_WORDS, debug=False):

    # Fetch the data with return_X_y=True returns a tuple (data, labels)
    train_data, train_labels = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), return_X_y=True)
    test_data, test_labels = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), return_X_y=True)

    # Get the class names from the 20 Newsgroups dataset
    class_names = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).target_names
    
    # Now, train_data and test_data contain the raw text
    train_texts = train_data
    test_texts = test_data

    if (debug):
        # Now, train_texts and test_texts can be fed into a tokenizer.
        print("train_texts:", type(train_texts), len(train_texts))
        print("train_texts[0]:", type(train_texts[0]), train_texts[0])

        print("test_texts:", type(test_texts), len(test_texts))
        print("test_texts[0]:", type(test_texts[0]), test_texts[0])

    # Convert labels to numpy.int64
    train_labels = train_labels.tolist()
    test_labels = test_labels.tolist()

    if (debug):
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

    if (debug):
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
    tokenizer = Tokenizer(num_words=max_words) 
    tokenizer.fit_on_texts(train_texts) 
    train_sequences = tokenizer.texts_to_sequences(train_texts) #Converting text to a vector of word indexes 
    test_sequences = tokenizer.texts_to_sequences(test_texts) 
    word_index = tokenizer.word_index 
    print('Found %s unique tokens.' % len(word_index))

    if (debug):
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

    if (debug):
        print("trainvalid_data:", type(trainvalid_data), trainvalid_data.shape)                         #Will print a 2D tensor
        print("trainvalid_data[0]:", type(trainvalid_data[0]), trainvalid_data[0].shape)                #Will print a 1D tensor
        print("trainvalid_data[0]:", trainvalid_data[0])                                                #Will print a 1D tensor with values as indexes

        print("test_data:", type(test_data), test_data.shape)                                   #Will print a 2D tensor
        print("test_data[0]:", type(test_data[0]), test_data[0].shape)                          #Will print a 1D tensor
        print("test_data[0]:\n", test_data[0])                                                  #Will print a 1D tensor with values as indexes

    # split the training data into a training set and a validation set
    indices = np.arange(trainvalid_data.shape[0])

    if (debug):
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

    x_test = test_data

    if (debug):
        #This is the data we will use for CNN and RNN training
        print("x_train:", type(x_train), x_train.shape)
        print("y_train:", type(y_train), y_train.shape)
        
        print("x_val:", type(x_val), x_val.shape)
        print("y_val:", type(y_val), y_val.shape)
        
        print("x_test:", type(x_test), x_test.shape)
        print("y_test:", type(y_test), y_test.shape)

    return x_train, y_train, train_labels, x_val, y_val, test_data, y_test, test_labels, class_names, word_index



def prep_nn_data(train_texts, train_labels, test_texts, test_labels, max_words=MAX_WORDS, max_sequence_length=MAX_SEQUENCE_LENGTH, debug=False):

    print(f'preparing data.... max_words={max_words}, max_sequence_length={max_sequence_length}')

    if (debug):
        print("train_texts:", type(train_texts), len(train_texts))
        print("train_texts[0]:", type(train_texts[0]), len(train_texts[0]))
        print("train_texts[0]:", train_texts[0])

        print("train_labels:", type(train_labels), len(train_labels))
        print("train_labels[0]:", type(train_labels[0]), train_labels[0].shape)
        print("train_labels[0]:", train_labels[0])

        print("test_texts:", type(test_texts), len(test_texts))
        print("test_texts[0]:", type(test_texts[0]), len(test_texts[0]))
        print("test_texts[0]:", test_texts[0])

        print("test_labels:", type(test_labels), len(test_labels))
        print("test_labels[0]:", type(test_labels[0]), test_labels[0].shape)
        print("test_labels[0]:", test_labels[0])

    # Vectorize these text samples into a 2D integer tensor using Keras Tokenizer 
    # Tokenizer is fit on training data only, and that is used to tokenize both train and test data. 
    tokenizer = Tokenizer(num_words=max_words) 
    tokenizer.fit_on_texts(train_texts) 

    train_sequences = tokenizer.texts_to_sequences(train_texts)                     #Converting text to a vector of word indexes 
    test_sequences = tokenizer.texts_to_sequences(test_texts) 

    word_index = tokenizer.word_index 
    
    if (debug):
        print("train_sequences:", type(train_sequences), len(train_sequences))              #This is a list of lists, one list for each review
        print("train_sequences[0]:", type(train_sequences[0]), len(train_sequences[0]))     #This is a list of word indexes for the first review
        print("train_sequences[0]:", train_sequences[0])                                    #This will print a list of word indexes (depends on the tokenizer)

        print("test_sequences:", type(test_sequences), len(test_sequences))                 #This is a list of lists, one list for each review
        print("test_sequences[0]:", type(test_sequences[0]), len(test_sequences[0]))        #This is a list of word indexes for the 25000th review
        print("test_sequences[0]:", test_sequences[0])                                      #This will print a list of word indexes (depends on the tokenizer)

        print("word_index:", type(word_index), len(word_index))                             #This is a dictionary of word_index. The words in our reviews are used as keys and the word indexes as values   

    print('Found %s unique tokens.' % len(word_index))

    # Converting this to sequences to be fed into neural network. 
    # initial padding of 0s, until vector is of size MAX_SEQUENCE_LENGTH
    trainvalid_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
    print("train_valid_data:", type(trainvalid_data), trainvalid_data.shape)
    print("train_valid_data[0]:", type(trainvalid_data[0]), len(trainvalid_data[0]))
    print("train_valid_data[0]:", trainvalid_data[0])
    
    trainvalid_labels = to_categorical(np.asarray(train_labels))
    print("train_valid_labels:", type(trainvalid_labels), trainvalid_labels.shape)
    print("train_valid_labels[0]:", type(trainvalid_labels[0]), len(trainvalid_labels[0]))
    print("train_valid_labels[0]:", trainvalid_labels[0])

    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print("test_data:", type(test_data), test_data.shape)
    print("test_data[0]:", type(test_data[0]), len(test_data[0]))
    print("test_data[0]:", test_data[0])

    test_labels = to_categorical(np.asarray(test_labels))
    print("test_labels:", type(test_labels), test_labels.shape)
    print("test_labels[0]:", type(test_labels[0]), len(test_labels[0]))
    print("test_labels[0]:", test_labels[0])

    # split the training data into a training set and a validation set
    indices = np.arange(trainvalid_data.shape[0])
    np.random.shuffle(indices)

    trainvalid_data = trainvalid_data[indices]
    trainvalid_labels = trainvalid_labels[indices]

    num_validation_samples = int(VALIDATION_SPLIT * trainvalid_data.shape[0])

    x_train = trainvalid_data[:-num_validation_samples]
    y_train = trainvalid_labels[:-num_validation_samples]
    x_val = trainvalid_data[-num_validation_samples:]
    y_val = trainvalid_labels[-num_validation_samples:]

    #This is the data we will use for CNN and RNN training

    print('Splitting the train data into train and valid is done')

    return x_train, y_train, x_val, y_val, test_data, test_labels, word_index



# Dense Neural Network Model
def build_dense_model(input_shape, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# CNN Model for text classification
def build_cnn_model(num_classes, max_words=MAX_WORDS, max_sequence_length=MAX_SEQUENCE_LENGTH):

    print(f'building cnn model with max_words = {max_words}, max_sequance_lenth = {max_sequence_length} and num_classes={num_classes}')

    # Define the CNN model
    cnnmodel = Sequential()

    # Force the Embedding layer to run on the CPU
    with tf.device('/CPU:0'):
        cnnmodel.add(Embedding(max_words, 128, input_length=max_sequence_length))

    # Rest of the model can run on the GPU
    cnnmodel.add(Conv1D(128, 5, activation='relu'))
    cnnmodel.add(MaxPooling1D(5))
    cnnmodel.add(Conv1D(128, 5, activation='relu'))
    cnnmodel.add(MaxPooling1D(5))
    cnnmodel.add(Conv1D(128, 5, activation='relu'))
    cnnmodel.add(GlobalMaxPooling1D())
    cnnmodel.add(Dense(128, activation='relu'))
    cnnmodel.add(Dense(num_classes, activation='softmax'))

    cnnmodel.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        #optimizer='adam'
        metrics=['acc']
        )
    
    return cnnmodel


# CNN Model for text classification
def build_cnn_model_bu(input_shape, num_classes):

    print(f'building cnn model with input_shape={input_shape}, num_classes={num_classes}')

    model = Sequential([
        Input(shape=(input_shape, 1)),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# LSTM Model for text classification
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape, 1)),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Attention-based Model for text classification
def build_attention_model(input_shape, num_classes):
    inputs = Input(shape=(input_shape,))
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    attention = Attention()([lstm_out, lstm_out])
    attention_out = tf.reduce_mean(attention, axis=1)
    outputs = Dense(num_classes, activation='softmax')(attention_out)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def predict(logits, classification_type='singlelabel'):

    if classification_type == 'multilabel':
        # Use NumPy for sigmoid and thresholding
        prediction = (1 / (1 + np.exp(-logits))) > 0.5

    elif classification_type == 'singlelabel':
        # Argmax for single-label classification
        prediction = np.argmax(logits, axis=1).reshape(-1, 1)

    else:
        print('unknown classification type')

    return prediction


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
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()





def decode_vectorized_text(vectorized_data, vectorizer):
    """
    Decodes vectorized input back into text using the vectorizer's vocabulary.

    Parameters:
    - vectorized_data: The input vectorized data (TF-IDF or similar format). Can be sparse or dense.
    - vectorizer: The fitted TfidfVectorizer or other vectorizer that was used to transform the text.

    Returns:
    - List of decoded text (original words).
    """
    # Get the vocabulary and reverse it (mapping indices back to words)
    vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    
    # Initialize list for decoded sentences
    decoded_sentences = []

    # Check if the data is sparse or dense
    if issparse(vectorized_data):
        # Convert to COO format for sparse matrices
        vectorized_data = vectorized_data.tocoo()

        # Iterate through each row of the sparse matrix
        for i in range(vectorized_data.shape[0]):
            # Get non-zero indices for the current row (this returns column indices)
            word_indices = vectorized_data.getrow(i).nonzero()[1]
            
            # Map the indices to words
            words = [vocab.get(idx, '') for idx in word_indices]
            
            # Join the words to form the original sentence
            decoded_sentence = ' '.join(words)
            decoded_sentences.append(decoded_sentence)
    else:
        # For dense matrices (numpy arrays)
        for row in vectorized_data:
            # Get indices where the value is non-zero
            word_indices = np.where(row > 0)[0]
            
            # Map the indices to words
            words = [vocab.get(idx, '') for idx in word_indices]
            
            # Join the words to form the original sentence
            decoded_sentence = ' '.join(words)
            decoded_sentences.append(decoded_sentence)

    return decoded_sentences





# Main function to train and evaluate the model
def main(args):

    print("\n\ttensorflow text classification with neural networks...")
    print("TensorFlow version:", tf.__version__)

    # Set the device
    device_name = set_device()
    print("Running on device:", device_name)

    with tf.device(device_name):

        if args.dataset == "reuters":

            print("Loading Reuters dataset...")

            print("num_words:", args.num_words)

            raw_train, x_train_vectorized, y_train, raw_test, x_test_vectorized, y_test, train_labels, test_labels, vectorizer = load_reuters_data(vtype=args.vtype, max_features=args.num_words, debug=args.debug)

            num_classes = 46
            print("num_classes:", num_classes)
            
            # Reuters class names (46 categories)
            class_names = [
                'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10',
                'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20',
                'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30',
                'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39', 'c40',
                'c41', 'c42', 'c43', 'c44', 'c45', 'c46'
            ]

            classification_type = 'multilabel'

        elif args.dataset == "20newsgroups":

            print("Loading 20 Newsgroups dataset...")

            if (args.model_type == 'dense'):

                raw_train, x_train_vectorized, y_train, raw_test, x_test_vectorized, y_test, train_labels, test_labels, class_names, \
                    vectorizer = load_20newsgroups_data(vtype=args.vtype, max_features=args.num_words, debug=args.debug)

                y_train = to_categorical(y_train, num_classes)
                y_test = to_categorical(y_test, num_classes)

                if (args.debug):
                    print("x_train_vectorized:", type(x_train_vectorized), x_train_vectorized.shape)
                    print("x_train_vectorized[0]:", x_train_vectorized[0])
                    print("y_train:", type(y_train), y_train.shape)
                    print("y_train[0]:", y_train[0])
                    print("train_labels:", type(train_labels), train_labels.shape)
                    print("train_labels[0]:", type(train_labels[0]), train_labels[0].shape)
                    print("train_labels[0]:", train_labels[0])

                    print("x_test_vectorized:", type(x_test_vectorized), x_test_vectorized.shape)
                    print("x_test_vectorized[0]:", x_test_vectorized[0])
                    print("y_test:", type(y_test), y_test.shape)
                    print("y_test[0]:", y_test[0])
                    print("test_labels:", type(test_labels), test_labels.shape)
                    print("test_labels[0]:", type(test_labels[0]), test_labels[0].shape)
                    print("test_labels[0]:", test_labels[0])

            elif (args.model_type == 'cnn'):

                x_train, y_train, train_labels, x_val, y_val, x_test, y_test, test_labels, class_names, \
                    word_index = load_20newsgroups_data_nn(max_words=args.max_words, debug=False)            

                if (args.debug):
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

                    print("x_test:", type(x_test), x_test.shape)
                    print("x_test[0]:", type(x_test[0]), x_test[0].shape)
                    print("y_test:", type(y_test), len(y_test))
                    print("y_test[0]:", y_test[0])
                    print("test_labels:", type(test_labels), len(test_labels))
                    print("test_labels[0]:", test_labels[0])

            num_classes = 20
            print("num_classes:", num_classes)
            
            classification_type = 'singlelabel'
            print("classification_type:", classification_type)

        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        if (args.debug):
            print("\ndataset:", args.dataset)
            print("classification_type:", classification_type)
            print("num_classes:", num_classes)
            print("class_names:", class_names)

            


        """
        # Select model architecture based on user input
        def select_model(model_type, input_shape, num_classes):
            if model_type == 'dense':
                return build_dense_model(input_shape, num_classes)
            elif model_type == 'cnn':
                return build_cnn_model(input_shape, num_classes)
            elif model_type == 'lstm':
                return build_lstm_model(input_shape, num_classes)
            elif model_type == 'attention':
                return build_attention_model(input_shape, num_classes)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        model = select_model(
            model_type=args.model_type, 
            input_shape=x_train.shape[1],
            num_classes=num_classes
        )
        """


        if (args.model_type == 'dense'):
        
            print("...building and fitting dense model...")

            # split based on a fixed percentage (e.g., 20% validation data)
            x_train_vectorized, x_val_vectorized, y_train, y_val = train_test_split(x_train_vectorized, y_train, test_size=TRAIN_TEST_SPLIT, random_state=42)

            if (args.debug):
                print("x_val_vectorized:", type(x_val_vectorized), x_val_vectorized.shape)
                print("x_val_vectorized[0]:", x_val_vectorized[0])
                print("y_val:", type(y_val), y_val.shape)
                print("y_val[0]:", y_val[0])
        
                decoded_vectorized_train_text = decode_vectorized_text(x_train_vectorized, vectorizer)
                print("decoded_vectorized_train_text:", type(decoded_vectorized_train_text), len(decoded_vectorized_train_text))
                print("decoded_vectorized_train_text[0]:", decoded_vectorized_train_text[0])

            model = build_dense_model(x_train_vectorized.shape[1], num_classes)
            print("model.summary():", model.summary())

            x_train = x_train_vectorized
            x_val = x_val_vectorized
            x_test = x_test_vectorized

        elif (args.model_type == 'cnn'):

            print("...building and fitting cnn model...")

            model = build_cnn_model(num_classes, max_words=args.max_words, max_sequence_length=MAX_SEQUENCE_LENGTH)

            if (args.debug):
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

                print("x_test:", type(x_test), x_test.shape)
                print("x_test[0]:", type(x_test[0]), x_test[0].shape)
                print("y_test:", type(y_test), len(y_test))
                print("y_test[0]:", y_test[0])
                print("test_labels:", type(test_labels), len(test_labels))
                print("test_labels[0]:", test_labels[0])


            print("model.summary():", model.summary())

        elif (args.model_type == 'lstm'):

            print("...building and fitting lstm model...")

            pass

        elif (args.model_type == 'attn'):

            print("...building and fitting attn model...")

            pass

        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # Callbacks for learning rate refinement, early stopping, and F1 score tracking
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        #f1_callback = F1ScoreCallback(validation_data=(x_test, y_test))             # Custom F1 score callback
        f1_callback = F1ScoreCallback(validation_data=(x_test, y_test))             # Custom F1 score callback

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        print("class_weights_dict:", class_weights_dict)

        print("\n\t ### training model... ###")

        # Then pass class_weights_dict to the fit function
        history = model.fit(x_train,
                            y_train,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            validation_data=(x_val, y_val),
                            callbacks=[reduce_lr, early_stop, f1_callback],
                            class_weight=class_weights_dict)
        
        if (args.plot):
            plot_history(history)

        print("evaluate model on test data...")
        results = model.evaluate(x_test, y_test)
        print("loss, accuracy:", results)

        test_preds = model.predict(x_test)
        
        # Thresholding for multi-label classification
        test_pred_classes = (test_preds > .5).astype(int)
        
        # Classification report with class labels and four decimal places
        report = classification_report(y_test, test_pred_classes, target_names=class_names, digits=4)
        print("\nClassification Report:\n", report)

        # Calculate macro and micro F1 scores
        macro_f1 = f1_score(y_test, test_pred_classes, average='macro')
        micro_f1 = f1_score(y_test, test_pred_classes, average='micro')
        
        # Log F1 scores
        print(f"\n\tMacro F1 Score = {macro_f1:.4f}, Micro F1 Score = {micro_f1:.4f}")


       

        




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a model on the Reuters or 20 Newsgroups dataset")

    parser.add_argument("--dataset", type=str, default="reuters", choices=["reuters", "20newsgroups"], help="Dataset to use (reuters or 20newsgroups)")
    parser.add_argument("--model_type", type=str, default="dense", choices=["dense", "cnn", "lstm", "attn"], help="Type of model to use (dense, cnn, lstm, attn or cnn)")
    parser.add_argument("--epochs", type=int, default=55, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--max_words", type=int, default=MAX_WORDS, help="Number of words to consider in the vocabulary")
    parser.add_argument("--vtype", type=str, default='count', help="Type of vectorization technique, either count or tfidf")
    parser.add_argument("--plot", action='store_true', default=False, help="Plot loss and accurracy history graphs for model training. Defaults to False, no plotting.")
    parser.add_argument("--debug", action='store_true', default=False, help="True if want to see debug output, False otherwise")

    args = parser.parse_args()

    main(args)