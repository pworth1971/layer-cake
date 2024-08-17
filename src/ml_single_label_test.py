
import pandas as pd
import re
import string

import nltk
from nltk.stem.wordnet import WordNetLemmatizer


from timeit import default_timer as timer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import model_selection, svm, metrics
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score, precision_score, hamming_loss, jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

from pprint import pprint
import argparse

import re
import string

import cufflinks as cf
import plotly.offline as pyo
import plotly.graph_objs as go

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pickle


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')

#
# we assume everything runs from bin directory
#
PICKLE_DIR = '../pickles/'
OUT_DIR = '../out/'
DATASET_DIR = '../datasets/'


#dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1'}
dataset_available = {'20newsgroups', 'bbc-news'}


def load_data(dataset='20newsgroups'):

    print(f"Loading data set {dataset}...")

    if (dataset == '20newsgroups'):
        
        print("Loading 20 newsgroups dataset...")

        # Fetch the 20 newsgroups dataset
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

        # Create a DataFrame from the Bunch object
        df = pd.DataFrame({
            'text': newsgroups.data,
            'category': newsgroups.target
        })

        # Add category names
        df['category_name'] = [newsgroups.target_names[i] for i in df['category']]

        print(f"Number of documents: {len(df)}")
        print(f"Number of categories: {len(df['category'].unique())}")
        print(f"Number of category names: {len(df['category_name'].unique())}")
        #pprint(list(df.target_names))
        #pprint(list(df.category_name))

        #df = df['category'].unique()    

        missing_values_df = missing_values(df)
        print(f"missing values:", missing_values_df)

        ### Start of Text Pre-processing
        print("preproccessing...")

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        string.punctuation

        ### 2. To LowerCase

        df['CleanedText'] = (df.text.apply(lambda x: x.lower()))

        ### 3. Removing Numbers and Special Characters including XXXXXX

        df['CleanedText'] =  (df.CleanedText.apply(lambda x: re.sub('\W+', ' ', x)))
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')

        df['CleanedText'] =  (df.CleanedText.apply(lambda x: re.sub(regex, '', x)))
        df['CleanedText'] =  (df.CleanedText.apply(lambda x: re.sub('xxxx', '', x)))
        df['CleanedText'] =  (df.CleanedText.apply(lambda x: re.sub('xx', '', x)))

        print("removing punctuation...")

        df['CleanedText'] =  (df.CleanedText.apply(lambda x: remove_punctuation(x)))

        ### 5. Tokenization
        #data['TokenizedText'] =  (data.CleanedText.apply(lambda x: re.split('W+',x)))

        print("removing stopwords...")
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        stopwords = set(stopwords.words("english"))
        df['CleanedText'] = df.CleanedText.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
        print("Stopwords removed")
        #print(df['CleanedText'][0])

        ## TFIDF already tokenizes the text so no need to tokenize it here
        # from nltk.tokenize import sent_tokenize, word_tokenize
        # data2['TokenizedText'] = data2.CleanedText.apply(word_tokenize)


        ### 7. Text Normalization  [Lemmatization] -->better than Stemming since it returns actual words
        ## lemmatization is an intelligent operation that uses dictionaries

        print("Lemmatizing...")

        df['LemmatizedText'] = lemmatization(df['CleanedText'])

        print("Lemmatized")
        #print(df['CleanedText'][0])
        #print(df['LemmatizedText'][0])

        print("Tokenizing...")
        
        # Tokenize the text data
        df['tokenized'] = df['text'].str.lower().apply(nltk.word_tokenize)

        #return df, df['category'].unique()
        return df
    
    elif (dataset == 'bbc-news'):

        print("Loading BBC News dataset...")

        for dirname, _, filenames in os.walk(DATASET_DIR+'bbc-news'):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        train_set = pd.read_csv(DATASET_DIR+'bbc-news/BBC News Train.csv')
        test_set = pd.read_csv(DATASET_DIR+'bbc-news/BBC News Test.csv')

        print("train_set:", train_set.shape)
        print(train_set.head())
        print("test_set:", test_set.shape)
        print(test_set.head())

        target_category = train_set['Category'].unique()
        print("target categories:", target_category)

        train_set['categoryId'] = train_set['Category'].factorize()[0]
        
        category = train_set[["Category","categoryId"]].drop_duplicates().sort_values('categoryId')
        print("after de-duping:", category)

        print(train_set.groupby('Category').categoryId.count())

        text = train_set["Text"] 
        print("text:\n", text.head())

        category = train_set["Category"]
        print("categories:\n", category.head())

        print("preprocessing...")
        train_set['Text'] = train_set['Text'].apply(preprocessDataset)
        text = train_set['Text']
        category = train_set['Category']
        print("text:\n", type(text), text.shape)
        print(text.head())

        #return text, category
        return train_set
    else:
        print(f"Dataset '{dataset}' not available.")
        return None


# --------------------------------------------------------------------------------------------------------------
# Core processing function
# --------------------------------------------------------------------------------------------------------------
def classify(dataset='20newsgrouops', args=None):
    
    if (args is None):
        print("No arguments passed.")
        return
    
    if (args.dataset not in dataset_available):
        print(f"Dataset '{args.dataset}' not available.")
        return
    
    pickle_file_name=f'{dataset}_{args.mode}_tokenized.pickle'

    print(f"Classifying {dataset}...")

    print(f"Loading data set {dataset}...")

    # Define the path to the pickle file
    pickle_file = PICKLE_DIR + pickle_file_name

    if os.path.exists(pickle_file):                                         # if the pickle file exists
        
        print(f"Loading tokenized data from '{pickle_file}'...")
        
        if (dataset == '20newsgroups'):
            # Initialize an empty DataFrame with the desired columns
            columns = ['tokenized', 'CleanedText', 'LemmatizedText', 'category', 'category_name', 'text']
            
            df = pd.DataFrame(columns=columns)
            
            # Load the data from the pickle file into the DataFrame
            with open(pickle_file, 'rb') as f:
                df = pickle.load(f)

            #categories = df['category'].unique()
            
        elif (dataset == 'bbc-news'):

            columns = []

            df = pd.DataFrame(columns=columns)
            
            with open(pickle_file, 'rb') as f:
                df = pickle.load(f)

            
    else:
        print(f"'{pickle_file}' not found, retrieving and preprocessing data set {dataset}...")

        #df, categories = load_data(dataset)
        df = load_data(dataset)

        print("df:", df.shape)
        print(df.head())

        # Save the tokenized DataFrame to a pickle file
        if (dataset == '20newsgroups'):
            with open(pickle_file, 'wb') as f:
                pickle.dump(df[['tokenized', 'CleanedText', 'LemmatizedText', 'category', 'category_name', 'text']], f)

        elif (dataset == 'bbc-news'):
            with open(pickle_file, 'wb') as f:
                pickle.dump(df[['ArticleId', 'Text', 'Category']], f)
                #pickle.dump(df, f)

    print("Tokenized data loaded, df:", df.shape)
    print(df.head())
    #print("categories:", categories)

    print("Processing data...")

    if (dataset == '20newsgroups'):

        """
        # POS Tagging and Counting
        tagged_titles = df['text'].apply(lambda x: nltk.pos_tag(nltk.word_tokenize(x)))

        def count_tags(title_with_tags):
            tag_count = {}
            for word, tag in title_with_tags:
                tag_count[tag] = tag_count.get(tag, 0) + 1
            return tag_count

        # Create a DataFrame with POS tag counts
        tagged_titles_df = pd.DataFrame(tagged_titles.apply(lambda x: count_tags(x)).tolist()).fillna(0)

        # Sum the occurrences of each tag across all documents
        tagged_titles_sum = tagged_titles_df.sum().sort_values(ascending=False)

        # Plot POS Tag Frequency
        trace = go.Bar(x=tagged_titles_sum.index, y=tagged_titles_sum.values)
        layout = go.Layout(title='Frequency of POS Tags in IT Support Tickets Dataset', xaxis=dict(title='POS'), yaxis=dict(title='Count'))
        fig = go.Figure(data=[trace], layout=layout)

        # This will open the plot in the default web browser
        pyo.plot(fig, filename='../../out/pos_tag_frequency.html')
        """

        # Feature Extraction and Model Training
        print("Splitting the dataset...")

        X = df['CleanedText']
        y = df['category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

        print("Vectorizing...")
        vectorizer = TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, use_idf=True)
        X_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        print("Fitting the model...")

        print("X_tfidf:", X_tfidf.shape)
        print("y_train:", y_train.shape)

        svm = LinearSVC(class_weight='balanced', max_iter=1000)
        clf = svm.fit(X_tfidf, y_train)

        print("Predicting...")
        print("X_test_tfidf:", X_test_tfidf.shape)
        print("y_test:", y_test.shape)

        y_pred = model_selection.cross_val_predict(svm, X_test_tfidf, y_test, cv=10)
        print("Accuracy for SVM :", metrics.accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        print("Using GridSearchCV...")
        svm_classifier = LinearSVC(class_weight='balanced', max_iter=1000)

        parameter_grid = {
            'class_weight': [None, 'balanced'],
            'C': np.logspace(-3, 3, 7)
            }

        cross_validation = StratifiedKFold()

        #scoring = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted']
        #from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score, precision_score, hamming_loss, jaccard_score, 

        scorers = {
                    'accuracy_score': make_scorer(accuracy_score),
                    'f1_score': make_scorer(f1_score, average='micro'),
                    'recall_score': make_scorer(recall_score, average='micro'),
                    'precision_score': make_scorer(precision_score, average='micro'),
                    'hamming_loss': make_scorer(hamming_loss),
                    'jaccard_score': make_scorer(jaccard_score, average='micro')
                }

        grid_search = GridSearchCV(
            n_jobs=-1, 
            estimator=svm_classifier,
            refit='f1_score',
            param_grid=parameter_grid,
            cv=cross_validation,
            #scoring=scoring
            scoring=scorers,
            return_train_score=True         # ensure train scores are calculated
            )

        grid_search.fit(X_tfidf, y_train)

        print('Best parameters: {}'.format(grid_search.best_params_))
        print("best_estimator:", grid_search.best_estimator_)
        print('Best score: {}'.format(grid_search.best_score_))
        print("cv_results_:", grid_search.cv_results_)

        results = grid_search.cv_results_

        print("Plotting the results...")

        # Define the metrics we want to plot
        metrics_to_plot = ['accuracy_score', 'f1_score', 'recall_score', 'precision_score', 'hamming_loss']

        # Iterate over each metric to create a separate plot
        for metric in metrics_to_plot:
            traces = []

            print(f"Plotting {metric}...")

            for sample in ["train", "test"]:

                key_mean = f"mean_{sample}_{metric}"
                key_std = f"std_{sample}_{metric}"

                print(f"Plotting {key_mean}...")
                print(f"Plotting {key_std}...")

                # Directly use the keys without conditional check
                sample_score_mean = np.nan_to_num(np.array(results[key_mean]) * 100)  # Convert to percentage and handle NaN
                sample_score_std = np.nan_to_num(np.array(results[key_std]) * 100)  # Convert to percentage and handle NaN

                x_axis = np.linspace(0, 100, len(sample_score_mean))

                # Create the trace for Plotly
                traces.append(
                    go.Scatter(
                        x=x_axis,
                        y=sample_score_mean,
                        mode='lines+markers',
                        name=f"{metric} ({sample})",
                        line=dict(dash='dash' if sample == 'train' else 'solid'),
                        error_y=dict(
                            type='data',
                            array=sample_score_std,
                            visible=True
                        ),
                        hoverinfo='x+y+name'
                    )
                )

            # Define the layout of the plot
            layout = go.Layout(
                title={'text': f"Training and Test Scores for {metric.capitalize()}",
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                xaxis=dict(title="Training Sample Percentage (%)"),
                yaxis=dict(title="Score (%)", range=[0, 100]),
                hovermode='closest'
            )

            # Create the figure
            fig = go.Figure(data=traces, layout=layout)

            # Write the plot to an HTML file
            filename = f'{OUT_DIR}training_test_scores_{metric}.html'
            pyo.plot(fig, filename=filename)

            print(f"Saved plot for {metric} as {filename}")

        # Extract the best estimator from the GridSearchCV
        best_model = grid_search.best_estimator_

        # Predict on the test set using the best model
        y_pred_best = best_model.predict(X_test_tfidf)

        print("Accuracy best score:", metrics.accuracy_score(y_test, y_pred_best))
        print(classification_report(y_test, y_pred_best))

        create_confusion_matrix(
            y_test, 
            y_pred_best, 
            title=f'Confusion Matrix for Best Model ({grid_search.best_params_})', 
            file_name=OUT_DIR+'svm_20newsgroups_confusion_matrix_best_model_table.png', 
            debug=False
            )
    
    elif (dataset == 'bbc-news'):

        print(f'classifying {dataset}...')

        print("Splitting the dataset...")
        
        #X_train, X_test, Y_train, Y_test = train_test_split(text,category, test_size = 0.3, random_state = 60,shuffle=True, stratify=category)
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            df['Text'],
            df['Category'], 
            test_size = 0.3, 
            random_state = 60,
            shuffle=True, 
            stratify=df['Category']
            )

        print("X_train:", type(X_train), X_train.shape)
        print("X_test:", type(X_test), X_test.shape)
        
        print("Y_train:", type(Y_train), Y_train.shape)
        print("Y_test:", type(Y_test), Y_test.shape)

        print("Naive Bayes Classifier")
        nb = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
            ])
        
        nb.fit(X_train,Y_train)

        test_predict = nb.predict(X_test)

        train_accuracy = round(nb.score(X_train,Y_train)*100)
        test_accuracy =round(accuracy_score(test_predict, Y_test)*100)

        print("Naive Bayes Train Accuracy Score : {}% ".format(train_accuracy ))
        print("Naive Bayes Test Accuracy Score  : {}% ".format(test_accuracy ))
        #print(classification_report(test_predict, Y_test, target_names=categories))
        print(classification_report(test_predict, Y_test))

        print("Decision Tree Classifier")
        dt = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('dt', DecisionTreeClassifier())
            ])

        dt.fit(X_train, Y_train)

        test_predict = dt.predict(X_test)

        train_accuracy = round(dt.score(X_train,Y_train)*100)
        test_accuracy =round(accuracy_score(test_predict, Y_test)*100)

        print("Decision Tree Train Accuracy Score : {}% ".format(train_accuracy ))
        print("Decision Tree Test Accuracy Score  : {}% ".format(test_accuracy ))
        #print(classification_report(test_predict, Y_test, target_names=categories))
        print(classification_report(test_predict, Y_test))

        print("Random Forest Classifier")
        rfc = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('rfc', RandomForestClassifier(n_estimators=100))
            ])

        rfc.fit(X_train, Y_train)

        test_predict = rfc.predict(X_test)

        train_accuracy = round(rfc.score(X_train,Y_train)*100)
        test_accuracy =round(accuracy_score(test_predict, Y_test)*100)

        print("K-Nearest Neighbour Train Accuracy Score : {}% ".format(train_accuracy ))
        print("K-Nearest Neighbour Test Accuracy Score  : {}% ".format(test_accuracy ))
        #print(classification_report(test_predict, Y_test, target_names=categories))
        print(classification_report(test_predict, Y_test))
    

    

def create_confusion_matrix(y_test, y_pred, title, file_name=OUT_DIR+'svm_20newsgroups_confusion_matrix_best_model_table.png', debug=False):

    print("creating confusion matrix...")

    # Assuming y_test and y_pred_best are already defined
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plotting the confusion matrix as a table with numbers
    fig, ax = plt.subplots(figsize=(10, 7))

    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    table = ax.table(
        cellText=conf_matrix,
        rowLabels=[f'Actual {i}' for i in range(conf_matrix.shape[0])],
        colLabels=[f'Predicted {i}' for i in range(conf_matrix.shape[1])],
        cellLoc='center',
        loc='center'
    )

    # Adjust the font size and layout
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)

    #plt.title(f'Confusion Matrix for Best Model ({grid_search.best_params_})', fontsize=18, pad=20)
    plt.title(title, fontsize=18, pad=20)

    # Save the plot to a file
    #confusion_matrix_filename = OUT_DIR+'svm_20newsgroups_confusion_matrix_best_model_table.png'
    confusion_matrix_filename = file_name
    plt.savefig(confusion_matrix_filename)
    plt.show()

    print(f"Confusion matrix saved as {confusion_matrix_filename}")

    accuracy = accuracy_score(y_test, y_pred)

    # Plain text explanation of the confusion matrix

    if (debug):
        print("\nHow to read this confusion matrix:")
        print("------------------------------------------------------")
        print("The confusion matrix shows the performance of the classification model.")
        print("Each row of the matrix represents the actual classes, while each column represents the predicted classes.")
        print("Values on the diagonal (from top-left to bottom-right) represent correct predictions (true positives and true negatives).")
        print("Values outside the diagonal represent incorrect predictions (false positives and false negatives).")
        print("\nAccuracy Score: {:.2f}%".format(accuracy * 100))
        
        print("\nConfusion Matrix Values:")
        for i in range(len(conf_matrix)):
            print(f"Actual class {i}:")
            for j in range(len(conf_matrix[i])):
                print(f"  Predicted as class {j}: {conf_matrix[i][j]}")

    print("Done.")

# --------------------------------------------------------------------------------------------------------------
#
# Utility functions for preprocessing data
#
# --------------------------------------------------------------------------------------------------------------
def missing_values(df):
    """
    Calculate the percentage of missing values for each column in a DataFrame.
    
    Args:
    df (pd.DataFrame): The input DataFrame to analyze.
    
    Returns:
    pd.DataFrame: A DataFrame containing the total count and percentage of missing values for each column.
    """
    # Calculate total missing values and their percentage
    total = df.isnull().sum()
    percent = (total / len(df) * 100)
    
    # Create a DataFrame with the results
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # Sort the DataFrame by percentage of missing values (descending)
    missing_data = missing_data.sort_values('Percent', ascending=False)
    
    # Filter out columns with no missing values
    missing_data = missing_data[missing_data['Total'] > 0]
    
    print("Columns with missing values:")
    print(missing_data)
    
    return missing_data


def remove_punctuation(x):
    punctuationfree="".join([i for i in x if i not in string.punctuation])
    return punctuationfree


# Function to lemmatize text with memory optimization
def lemmatization(texts, chunk_size=1000):
    lmtzr = WordNetLemmatizer()
    
    num_chunks = len(texts) // chunk_size + 1
    #print(f"Number of chunks: {num_chunks}")
    for i in range(num_chunks):
        chunk = texts[i*chunk_size:(i+1)*chunk_size]
        texts[i*chunk_size:(i+1)*chunk_size] = [' '.join([lmtzr.lemmatize(word) for word in text.split()]) for text in chunk]
    
    return texts


def preprocessDataset(train_text):
    # Ensure input is string
    train_text = str(train_text)
    
    # Word tokenization using NLTK's word_tokenize
    tokenized_train_set = word_tokenize(train_text.lower())
    
    # Stop word removal
    stop_words = set(stopwords.words('english'))
    stopwordremove = [i for i in tokenized_train_set if i not in stop_words]
    
    # Join words into sentence
    stopwordremove_text = ' '.join(stopwordremove)
    
    # Remove numbers
    numberremove_text = ''.join(c for c in stopwordremove_text if not c.isdigit())
    
    # Stemming using NLTK's PorterStemmer
    stemmer = PorterStemmer()
    stem_input = word_tokenize(numberremove_text)
    stem_text = ' '.join([stemmer.stem(word) for word in stem_input])
    
    # Lemmatization using NLTK's WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    lem_input = word_tokenize(stem_text)
    lem_text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in lem_input])
    
    return lem_text

# --------------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification Testing')
    
    parser.add_argument('--dataset', type=str, default='20newsgroups', metavar='N',
                        help=f'dataset, one in {dataset_available}')
    
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str',
                        help=f'path where to load the pickled dataset from')
    
    parser.add_argument('--log-file', type=str, default='../log/svm.test', metavar='N', help='path to the application log file')
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', 
                        help=f'learner (svm, lr, or nb)')
    
    parser.add_argument('--mode', type=str, default='tfidf', metavar='N',
                        help=f'mode, in [tfidf, count]')

    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext|bert|llama',
                        help='pretrained embeddings, use "glove", "word2vec", "fasttext", "bert", or "llama" (default None)')
                             
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the C parameter in the SVM')
    

    args = parser.parse_args()

    print("args:", type(args), args)

    classify(dataset=args.dataset, args=args)
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------