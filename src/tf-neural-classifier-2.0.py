
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, LSTM, Attention, Bidirectional, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


MAX_WORDS = 10000
TRAIN_TEST_SPLIT = 0.2


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

    

def load_reuters_data(vtype="count", max_features=MAX_WORDS):

    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=max_features)
    
    if (vtype == 'count'):
        vectorizer = CountVectorizer(max_features=max_features, lowercase=False)
    elif (vtype == 'tfidf'):
        vectorizer = TfidfVectorizer(max_features=max_features, lowercase=False)

    # Get the word index from the Reuters dataset
    word_index = reuters.get_word_index()
    reverse_word_index = {value: key for (key, value) in word_index.items()}

    # Convert the list of word indices back into readable text (documents)
    train_data = [' '.join([reverse_word_index.get(i - 3, '?') for i in seq]) for seq in train_data]
    test_data = [' '.join([reverse_word_index.get(i - 3, '?') for i in seq]) for seq in test_data]
        
    # Use vectorizer for feature extraction
    x_train = vectorizer.fit_transform(train_data).toarray()
    x_test = vectorizer.transform(test_data).toarray()

    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    
    return x_train, y_train, x_test, y_test, train_labels, test_labels


# Load and preprocess 20 Newsgroups data
def load_20newsgroups_data(vtype='count', max_features=MAX_WORDS):

    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    if (vtype == 'count'):
        vectorizer = CountVectorizer(max_features=max_features, lowercase=False)
    elif (vtype == 'tfidf'):
        vectorizer = TfidfVectorizer(max_features=max_features, lowercase=False)

    x_train_vectorized = vectorizer.fit_transform(newsgroups_train.data).toarray()
    x_test_vectorized = vectorizer.transform(newsgroups_test.data).toarray()

    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    return x_train_vectorized, y_train, x_test_vectorized, y_test, newsgroups_train.target, newsgroups_test.target

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
def build_cnn_model(input_shape, num_classes):
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

            x_train, y_train, x_test, y_test, train_labels, test_labels = load_reuters_data(vtype=args.vtype, max_features=args.num_words)

            if (args.debug):
                print("y_train:", type(y_train), y_train.shape)
                print("y_train[0]:", y_train[0])
                print("y_test:", type(y_test), y_test.shape)
                print("y_test[0]:", y_test[0])

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
            x_train, y_train, x_test, y_test, train_labels, test_labels = load_20newsgroups_data(vtype=args.vtype, max_features=args.num_words)

            if (args.debug):
                print("y_train:", type(y_train), y_train.shape)
                print("y_train[0]:", y_train[0])
                print("y_test:", type(y_test), y_test.shape)
                print("y_test[0]:", y_test[0])

            num_classes = 20
            print("num_classes:", num_classes)

            # Get the class names from the 20 Newsgroups dataset
            class_names = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')).target_names

            y_train = to_categorical(y_train, num_classes)
            y_test = to_categorical(y_test, num_classes)
            
            classification_type = 'singlelabel'

        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        if (args.debug):
            print("\ndataset:", args.dataset)
            print("classification_type:", classification_type)
            print("num_classes:", num_classes)
            print("class_names:", class_names)

            print("x_train:", type(x_train), x_train.shape)
            print("x_train[0]:", x_train[0])
            print("y_train:", type(y_train), y_train.shape)
            print("y_train[0]:", y_train[0])
            print("train_labels:", type(train_labels), train_labels.shape)
            print("train_labels:", train_labels)

            print("x_test:", type(x_test), x_test.shape)
            print("x_test[0]:", x_test[0])
            print("y_test:", type(y_test), y_test.shape)
            print("y_test[0]:", y_test[0])
            print("test_labels:", type(test_labels), test_labels.shape)
            print("test_labels:", test_labels)


        # New code to split based on a fixed percentage (e.g., 20% validation data)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        if (args.debug):
            print("x_val:", type(x_val), x_val.shape)
            print("x_val[0]:", x_val[0])
            print("y_val:", type(y_val), y_val.shape)
            print("y_val[0]:", y_val[0])

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
        print("-- model --:\n", model)

        # Callbacks for learning rate refinement, early stopping, and F1 score tracking
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        f1_callback = F1ScoreCallback(validation_data=(x_test, y_test))             # Custom F1 score callback

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        """
        model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        
        print(f"Test accuracy: {test_acc}")
        """

        # Then pass class_weights_dict to the fit function
        history = model.fit(x_train,
                            y_train,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            validation_data=(x_val, y_val),
                            callbacks=[reduce_lr, early_stop, f1_callback],
                            class_weight=class_weights_dict)
        
        plot_history(history)

        results = model.evaluate(x_test, y_test)
        print("model test loss, test accuracy:", results)

        test_preds = model.predict(x_test)
        #print("val_predictions:", type(val_predictions), val_predictions.shape)
        #print("val_predictions[0]:", val_predictions[0])

        # Thresholding for multi-label classification
        test_pred_classes = (test_preds > .5).astype(int)
        #print("val_pred_classes:", type(val_pred_classes), val_pred_classes.shape)
        #print("val_pred_classes[0]:", val_pred_classes[0])

        # Calculate macro and micro F1 scores
        macro_f1 = f1_score(y_test, test_pred_classes, average='macro')
        micro_f1 = f1_score(y_test, test_pred_classes, average='micro')
        
        # Log F1 scores
        print(f"\n\tMacro F1 Score = {macro_f1:.4f}, Micro F1 Score = {micro_f1:.4f}")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a model on the Reuters or 20 Newsgroups dataset")

    parser.add_argument("--dataset", type=str, default="reuters", choices=["reuters", "20newsgroups"], help="Dataset to use (reuters or 20newsgroups)")
    parser.add_argument("--model_type", type=str, default="dense", choices=["dense", "cnn", "lstm", "cnn", "attn"], help="Type of model to use (dense, cnn, lstm, attn or cnn)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--num_words", type=int, default=MAX_WORDS, help="Number of words to consider in the vocabulary")
    parser.add_argument("--vtype", type=str, default='count', help="Type of vectorization technique, either count or tfidf")
    parser.add_argument("--debug", action='store_true', default=False, help="True if want to see debug output, False otherwise")

    args = parser.parse_args()

    main(args)