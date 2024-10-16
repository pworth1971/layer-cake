import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from util.metrics import evaluation_nn


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


# Load and preprocess Reuters data
def load_reuters_data(num_words):
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)
    x_train = vectorize_sequences(train_data, dimension=num_words)
    x_test = vectorize_sequences(test_data, dimension=num_words)
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    return x_train, y_train, x_test, y_test, train_labels, test_labels


# Load and preprocess 20 Newsgroups data
def load_20newsgroups_data():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(max_features=10000)
    x_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
    x_test = vectorizer.transform(newsgroups_test.data).toarray()

    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    return x_train, y_train, x_test, y_test, newsgroups_train.target, newsgroups_test.target


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results


def build_dense_model(input_shape, output_classes, lr=1e-3):

    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(output_classes, activation="softmax")
    ])

    # Lower the learning rate to 1e-4
    model.compile(
        optimizer=Adam(learning_rate=lr), 
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def build_cnn_model(input_shape, output_classes, lr=1e-3):

    model = keras.Sequential([
        layers.Input(shape=(input_shape, 1)),
        layers.Conv1D(128, 5, activation="relu"),
        layers.MaxPooling1D(5),
        layers.Conv1D(128, 5, activation="relu"),
        layers.MaxPooling1D(5),
        layers.Conv1D(128, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(output_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=lr),       
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


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



def main(args):

    # Set the device
    device_name = set_device()

    with tf.device(device_name):

        if args.dataset == "reuters":
            x_train, y_train, x_test, y_test, train_labels, test_labels = load_reuters_data(args.num_words)
            num_classes = 46
            classification_type = 'multilabel'
        elif args.dataset == "20newsgroups":
            x_train, y_train, x_test, y_test, train_labels, test_labels = load_20newsgroups_data()
            num_classes = 20
            y_train = to_categorical(y_train, num_classes)
            y_test = to_categorical(y_test, num_classes)
            classification_type = 'singlelabel'
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        # Expand dimensions for CNN (adding channel dimension)
        if args.model_type == "cnn":
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)

        x_val = x_train[:1000]
        partial_x_train = x_train[1000:]
        y_val = y_train[:1000]
        partial_y_train = y_train[1000:]

        # Build either a Dense model or a CNN model based on user input
        if args.model_type == "cnn":
            model = build_cnn_model(input_shape=x_train.shape[1], output_classes=num_classes)
        else:
            model = build_dense_model(input_shape=x_train.shape[1], output_classes=num_classes)
        print("model:\n", model.summary())

        # Callbacks for learning rate refinement and early stopping
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        # Train model with callbacks
        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            validation_data=(x_val, y_val),
                            callbacks=[reduce_lr, early_stop])
        
        plot_history(history)

        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Use the evaluation method
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = evaluation_nn(
            y_true=test_labels, 
            y_pred=y_pred,
            classification_type=classification_type,
            debug=False
            )

        # Print the metrics with four decimal places
        print(f"Macro F1 Score: {Mf1}")
        print(f"Micro F1 Score: {mf1}")
        print(f"Accuracy: {accuracy}")
        print(f"Hamming Loss: {h_loss}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Jaccard Index: {j_index}")
        
        # Print classification report (sklearn) with four decimal places
        print("\nClassification Report:\n")
        print(classification_report(test_labels, y_pred_classes, digits=4))



if __name__ == "__main__":
    # Command-line argument parser

    parser = argparse.ArgumentParser(description="Train a model on the Reuters or 20 Newsgroups dataset")
    parser.add_argument("--dataset", type=str, default="reuters", choices=["reuters", "20newsgroups"], help="Dataset to use (reuters or 20newsgroups)")
    parser.add_argument("--model_type", type=str, default="dense", choices=["dense", "cnn"], help="Type of model to use (dense or cnn)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--num_words", type=int, default=10000, help="Number of words to consider in the vocabulary (for Reuters dataset)")

    args = parser.parse_args()
    main(args)
