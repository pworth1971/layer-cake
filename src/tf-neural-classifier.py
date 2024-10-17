import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping 
from tensorflow.keras import layers, Sequential
from tensorflow.keras import backend as K

from sklearn.metrics import f1_score, classification_report
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

from util.metrics import evaluation_nn

from scipy.sparse import csr_matrix
import torch

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



def dense_model(input_shape, output_classes, lr=1e-3):

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


def main(args):

    print("\n\ttensorflow text classification with neural networks...")

    print("TensorFlow version:", tf.__version__)

    # Set the device
    device_name = set_device()

    print("Device:", device_name)

    with tf.device(device_name):

        if args.dataset == "reuters":

            print("Loading Reuters dataset...")

            print("num_words:", args.num_words)

            x_train, y_train, x_test, y_test, train_labels, test_labels = load_reuters_data(args.num_words)

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
            x_train, y_train, x_test, y_test, train_labels, test_labels = load_20newsgroups_data()

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

        # Expand dimensions for CNN (adding channel dimension)
        if args.model_type == "cnn":
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)

        x_val = x_train[:1000]
        partial_x_train = x_train[1000:]
        y_val = y_train[:1000]
        partial_y_train = y_train[1000:]

        # Build either a Dense model or a CNN model based on user input
        if args.model_type == "dense":
            model = dense_model(input_shape=x_train.shape[1], output_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        print("model:\n", model.summary())

        # Callbacks for learning rate refinement, early stopping, and F1 score tracking
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        f1_callback = F1ScoreCallback(validation_data=(x_test, y_test))             # Custom F1 score callback

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Then pass class_weights_dict to the fit function
        history = model.fit(partial_x_train,
                            partial_y_train,
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

        """
        # predictions
        y_pred = model.predict(x_test)
        print("y_pred:", type(y_pred), y_pred.shape)
        print("y_pred[0]:", y_pred[0])

        yte_ = csr_matrix(predict(y_pred, classification_type=classification_type))
        print("yte_:", type(yte_), yte_.shape)
        print("yte_[0]:", yte_[0])

        # actuals (true values)
        print("y_test:", type(y_test), y_test.shape)
        print("y_test[0]:", y_test[0])

        if classification_type == 'multilabel':
            # Multi-label case: Use sigmoid and threshold at 0.5
            y_pred_classes = (yte_ > 0.5).astype(int)
        elif classification_type == 'singlelabel':
            # Single-label case: Use argmax to get class predictions
            y_pred_classes = np.argmax(yte_, axis=1)
        else:
            raise ValueError(f"Unsupported classification type: {classification_type}")
        
        print("y_pred_classes:", type(y_pred_classes), y_pred_classes.shape)
        print("y_pred_classes[0]:", y_pred_classes[0])

        #print("True labels:", type(test_labels), test_labels.shape)
        #print("True labels[0]:", test_labels[0])
        #print("Predictions:\n", type(y_pred_classes), y_pred_classes.shape)
        #print("Predictions[0]:", y_pred_classes[0])
        
        # Print the classification report
        if (classification_type == 'singlelabel'):

            print("\nClassification Report (single-label):\n")
            print(classification_report(y_test, yte_, target_names=class_names, digits=4))

        elif (classification_type == 'multilabel'):

            print("Classification Report (multi-label):\n")
            print(classification_report(y_test, yte_, target_names=class_names, digits=4))

        # Use the evaluation method
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = evaluation_nn(
            y_true=y_test, 
            #y_pred=y_pred_classes,
            y_pred=yte_,
            classification_type=classification_type,
            debug=False
            )
        
        # Print the metrics with four decimal places
        print(f"Macro F1 Score: {float(Mf1):.4f}")
        print(f"Micro F1 Score: {float(mf1):.4f}")
        print(f"Accuracy: {float(accuracy):.4f}")
        print(f"Hamming Loss: {float(h_loss):.4f}")
        print(f"Precision: {float(precision):.4f}")
        print(f"Recall: {float(recall):.4f}")
        print(f"Jaccard Index: {float(j_index):.4f}")
        """

        



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
