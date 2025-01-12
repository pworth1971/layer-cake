import numpy as np
import argparse
import os
from tqdm import tqdm
import pandas as pd
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from datasets import Dataset

import torch
from transformers import LlamaTokenizer, LlamaTokenizerFast, LlamaModel
from transformers import LlamaForSequenceClassification, Trainer, TrainingArguments
# from transformers.models import llama

import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')


# ---------------------------------------------------------------------------------------------------
# constants
#
DATASET_DIR = '../datasets/'
OUT_DIR = '../out'
LOG_DIR = '../log'
PICKLE_DIR = '../pickles'
VECTOR_CACHE = '../.vector_cache'

EPOCHS = 3
MAX_VOCAB_SIZE = 1000
TEST_SIZE = 0.2
BATCH_SIZE = 8

LLAMA_MODEL = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096
LLAMA_EMBEDDING_DIM = 4096
TOKEN_TOKENIZER_MAX_LENGTH = 512
# ---------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------
#
# tokens for LLAMA model access, must be requested from huggingface
# 
# must login to huggingface first 
# (see https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login) 
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'
# ---------------------------------------------------------------------------------------------------


def load_dataset(dataset = 'bbc_news'):
        
    print("loading dataset...")

    stop_words = set(stopwords.words('english'))

    #
    # BBC News 
    #
    if (dataset == 'bbc_news'):
        
        # Load the BBC News dataset
        print(f'\n\tloading BBC News dataset from {DATASET_DIR}...')

        for dirname, _, filenames in os.walk(DATASET_DIR + 'bbc-news'):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        # Load datasets
        train_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Train.csv')
        #test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')

        print("train_set:", train_set.shape)
        #print("test_set:", test_set.shape)    

        print("train_set columns:", train_set.columns)
        #print("train_set:\n", train_set.head())

        #train_set['Category'].value_counts().plot(kind='bar', title='Category distribution in training set')
        #train_set['Category'].value_counts()
        print("Unique Categories:\n", train_set['Category'].unique())
        numCats = len(train_set['Category'].unique())
        print("# of categories:", numCats)

        X_raw = train_set['Text'].tolist()
        y = np.array(train_set['Category'])

        target_names = train_set['Category'].unique()
    #
    # 20 Newsgroups
    #
    elif (dataset == '20newsgroups'):
        
        print(f'\n\tloading 20 Newsgroups dataset...')        

        # Load the 20 newsgroups dataset
        newsgroups_data = fetch_20newsgroups(subset='all')
        X_raw, y = newsgroups_data.data, newsgroups_data.target

        target_names = newsgroups_data.target_names

    print("X_raw:", type(X_raw), len(X_raw))
    #print("X_raw:", X_raw)
    print("y", type(y), len(y))
    print("target_names:", target_names)


    print("removing stopwords...")

    # Function to remove stopwords before tokenization
    def remove_stopwords(texts):
        filtered_texts = []
        for text in texts:
            filtered_words = [word for word in text.split() if word.lower() not in stop_words]
            filtered_texts.append(" ".join(filtered_words))
        return filtered_texts

    # Remove stopwords from the raw text
    X_raw = remove_stopwords(X_raw)
    print("X_raw:", type(X_raw), len(X_raw))

    print("X_raw[0]:\n", X_raw[0])
    print("y[0]:", y[0])

    # Split the dataset into training and testing sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=TEST_SIZE, random_state=42)

    print("X_train_raw:", type(X_train_raw), len(X_train_raw))
    print("X_test_raw:", type(X_test_raw), len(X_test_raw))
    print("y_train:", type(y_train), len(y_train))
    print("y_test:", type(y_test), len(y_test))

    return X_train_raw, X_test_raw, y_train, y_test, target_names



# ---------------------------------------------------------------------------------------------------------------------

def get_llama_vocab_embeddings(embeddings_file, tfidf_vectorizer, tokenizer, device):

    print("get_llama_vocab_embeddings...")

    print("embeddings_file:", embeddings_file)

    # Check if embeddings are already saved, if not, compute and save them
    if os.path.exists(embeddings_file):
        with open(embeddings_file, "rb") as f:
            llama_vocab_embeddings = pickle.load(f)
    else:
        # Create a vocabulary list of LLaMA encoded tokens based on the vectorizer vocabulary
        llama_vocab_embeddings = {}
        for token in tqdm(tfidf_vectorizer.get_feature_names_out(), desc="encoding vocabulary using LlaMa (pretrained) embeddings..."):
            input_ids = tokenizer.encode(token, return_tensors='pt').to(device)
            with torch.no_grad():
                output = model(input_ids)
            llama_vocab_embeddings[token] = output.last_hidden_state.mean(dim=1).cpu().numpy()

        with open(embeddings_file, "wb") as f:
            pickle.dump(llama_vocab_embeddings, f)

    return llama_vocab_embeddings


# ---------------------------------------------------------------------------------------------------------------------

def run_solo(X_train_tfidf, X_test_tfidf, y_train, y_test, target_names, vectorizer, llama_vocab_embeddings, vocab):

    print("running solo...")

    # Project the TF-IDF vectors into the LLaMA embedding space
    def llama_weighted_average_vectorization(tfidf_vectors, vocab_embeddings, vocab):
        print("converting tfidf vectorized data into llama embedding space...")
            
        print("tfidf_vectors:", type(tfidf_vectors), tfidf_vectors.shape)
        print("vocab_embeddings:", type(vocab_embeddings), len(vocab_embeddings))
        print("vocab:", type(vocab), vocab.shape)
        
        embedded_vectors = np.zeros((tfidf_vectors.shape[0], list(vocab_embeddings.values())[0].shape[1]))
        print("embedded_vectors:", type(embedded_vectors), embedded_vectors.shape)
        
        for i, doc in enumerate(tfidf_vectors):
            for j, token in enumerate(vocab):
                if token in vocab_embeddings:
                    embedded_vectors[i] += doc[j] * vocab_embeddings[token].squeeze()
        
        return embedded_vectors

    # convert the training and testing datasets
    vocab = vectorizer.get_feature_names_out()
    print("vocab (get_feature_names_out):", type(vocab), vocab.shape)

    vect_vocab = vectorizer.vocabulary_
    print("vect_vocab:", type(vect_vocab), len(vect_vocab))

    # Use the tokenizer's vocabulary directly, lowercased for consistency
    lower_vect_vocab = {k.lower(): v for k, v in vectorizer.vocabulary_.items()}
    print("lower_vect_vocab:", type(lower_vect_vocab), len(lower_vect_vocab))
            
    print("encoding dataset using LlaMa embeddings (weighted average approach)...")
            
    X_train_encoded_wa = llama_weighted_average_vectorization(X_train_tfidf, llama_vocab_embeddings, vocab)
    X_test_encoded_wa = llama_weighted_average_vectorization(X_test_tfidf, llama_vocab_embeddings, vocab)
    print("X_train_projected_wa:", type(X_train_encoded_wa), X_train_encoded_wa.shape)
    print("X_test_projected_wa:", type(X_test_encoded_wa), X_test_encoded_wa.shape)

    print("training SVM classifier...")

    run_svm_model(
        X_train_encoded_wa,
        X_test_encoded_wa,
        y_train,
        y_test,
        target_names,
        vectorizer
        )


# ---------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------

def run_dot(X_train_tfidf, X_test_tfidf, y_train, y_test, target_names, vectorizer, llama_vocab_embeddings, vocab):

    print("running dot...")

    # Function to convert llama_vocab_embeddings (dict) to a numpy matrix
    def convert_dict_to_matrix(vocab_embeddings, vocab):
        
        print("converting dict to matrix...")
        
        embedding_dim = LLAMA_EMBEDDING_DIM

        print("embedding_dim:", embedding_dim)
        
        # Assuming all embeddings have the same dimension and it's correctly 4096 as per the LLaMA model dimension
        embedding_matrix = np.zeros((len(vocab), embedding_dim))  # Shape (vocab_size, embedding_dim)

        print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)
        
        for i, token in enumerate(vocab):
            if token in vocab_embeddings:
                # Direct assignment of the embedding which is already in the correct shape (4096,)
                embedding_matrix[i, :] = vocab_embeddings[token]
            else:
                # Initialize missing tokens with zeros or a small random value
                embedding_matrix[i, :] = np.zeros(embedding_dim)

        return embedding_matrix

    # Function to project the TF-IDF vectors into the LLaMA embedding space using matrix multiplication
    def project_tfidf_to_llama(tfidf_vectors, embedding_matrix):
        return np.dot(tfidf_vectors, embedding_matrix)

    print("building llama vocabulary matrix for dataset vocab...")

    llama_vocab_matrix = convert_dict_to_matrix(llama_vocab_embeddings, vocab)
    print("llama_vocab_matrix:", type(llama_vocab_matrix), llama_vocab_matrix.shape)
    print("llama_vocab_matrix[0]:\n", llama_vocab_matrix[0])

    print("-- before numpy.dot operation...")
    print("X_train_tfidf:", type(X_train_tfidf), X_train_tfidf.shape)
    print("X_train_tfidf[0]:\n", X_train_tfidf[0])

    print("X_test_tfidf:", type(X_test_tfidf), X_test_tfidf.shape)
    print("X_test_tfidf[0]:\n", X_test_tfidf[0])

    # Project the training and testing sets
    X_train_projected_dot = project_tfidf_to_llama(X_train_tfidf, llama_vocab_matrix)
    X_test_projected_dot = project_tfidf_to_llama(X_test_tfidf, llama_vocab_matrix)

    print("-- after numpy.dot product operation (input to SVM)...")
    print("X_train_projected_dot:", type(X_train_projected_dot), X_train_projected_dot.shape)
    print("X_train_projected_dot[0]:\n", X_train_projected_dot[0])

    print("X_test_projected_dot:", type(X_test_projected_dot), X_test_projected_dot.shape)
    print("X_test_projected_dot[0]:\n", X_test_projected_dot[0])

    print("y_train:", type(y_train), y_train.shape)
    print("y_train[0]:", y_train[0])

    """
    # Train an SVM classifier on the projected features
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_projected_dot, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test_projected_dot)

    # Print classification report
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    """

    run_svm_model(
        X_train_projected_dot,
        X_test_projected_dot,
        y_train,
        y_test,
        category_names=target_names,
        vectorizer=vectorizer
        )



def run_svm_model(X_train, X_test, y_train, y_test, category_names, vectorizer, optimized=False, plot=False):

    print("Training default Support Vector Machine model...")
    
    default_pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('lr', LinearSVC(max_iter=1000))
    ])

    default_pipeline.fit(X_train, y_train)
    y_pred_default = default_pipeline.predict(X_test)

    print("\nDefault Support Vector Mechine Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
    print(classification_report(y_true=y_test, y_pred=y_pred_default, target_names=category_names, digits=4))

    if (optimized):

        # Optimize Support Vector Machine with GridSearchCV
        print("Optimizing Support Vector Machine model with GridSearchCV...")

        # Define the pipeline
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('svm', LinearSVC(max_iter=1000))
        ])

        # Define the parameter grid
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],     # Unigrams, bigrams, or trigrams
            'tfidf__use_idf': [True, False],                    # Whether to use IDF
            'tfidf__sublinear_tf': [True, False],               # Sublinear term frequency
            'svm__penalty': ['l1', 'l2'],                       # Regularization method
            'svm__loss': ['hinge', 'squared_hinge'],            # Loss function
            'svm__multi_class': ['ovr', 'crammer_singer'],      # Multi-class strategy
            'svm__class_weight': [None, 'balanced'],            # Class weights
            'svm__C': np.logspace(-3, 3, 7)                     # Regularization parameter   
        }

        print("param_grid:", param_grid)

        cross_validation = StratifiedKFold()

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
            estimator=pipeline,
            refit='f1_score',
            param_grid=param_grid,
            cv=cross_validation,
            #scoring=scoring
            scoring=scorers,
            return_train_score=True         # ensure train scores are calculated
            )

        # Fit the model
        grid_search.fit(X_train, y_train)

        print('Best parameters: {}'.format(grid_search.best_params_))
        print("best_estimator:", grid_search.best_estimator_)
        print('Best score: {}'.format(grid_search.best_score_))
        print("cv_results_:", grid_search.cv_results_)

        results = grid_search.cv_results_

        if (plot):

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
        y_pred_best = best_model.predict(X_test)

        print("Accuracy best score:", metrics.accuracy_score(y_test, y_pred_best))
        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=category_names, digits=4))
        
        return y_test, y_pred_best
    else:
        return y_test, y_pred_default



# ---------------------------------------------------------------------------------------------------------------------
# fine_tune()
# ---------------------------------------------------------------------------------------------------------------------

def fine_tune(train_texts, test_texts, train_labels, test_labels, num_labels, device):

    print("fine tuning....")

    # Step 1: Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels})

    print("\n\tbuilding the model...")

    model = LlamaForSequenceClassification.from_pretrained(
        LLAMA_MODEL,
        num_labels=num_labels,
        cache_dir=VECTOR_CACHE + '/LLaMa'
    ).to(device)

    # Load the pre-trained LLaMA Fast tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained(
        LLAMA_MODEL,
        cache_dir=VECTOR_CACHE + '/LLaMa'
    )

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
#        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
 #       model.resize_token_embeddings(len(tokenizer))  # Resize model's embeddings to accommodate the new token

    # set the pad token of the model's configuration
    model.config.pad_token_id = model.config.eos_token_id

    print("model:\n ", model)
    print("tokenizer:\n", tokenizer)

    # Custom tokenizer function using LLaMA tokenizer
    def llama_tokenize_function(text):

        #print("text:", type(text), text)

        # Tokenize the text with padding and truncation
        tokens = tokenizer(text=text['text'], text_target=text['labels'], padding="max_length", truncation=True, max_length=TOKEN_TOKENIZER_MAX_LENGTH)
        tokens["labels"] = text["labels"]  # Ensure labels are passed
        return tokens

    # Apply tokenization
    train_dataset = train_dataset.map(llama_tokenize_function, batched=True)
    test_dataset = test_dataset.map(llama_tokenize_function, batched=True)

    # Remove unnecessary columns and set the dataset format to PyTorch
    #train_dataset = train_dataset.remove_columns(["text"])
    #test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    """
    # Use DataLoader for smaller batches
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    """

    # Step 4: Define training arguments (with gradient accumulation and mixed precision)
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=LOG_DIR,
        #fp16=True,                                  # Enable mixed precision for memory efficiency, only supported on GPU
        gradient_accumulation_steps=4,              # Accumulate gradients over 4 steps to simulate larger batch size
    )

    # Step 5: Define metrics for evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(logits, dim=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = (predictions == labels).float().mean().item()
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Step 6: Initialize Trainer and start fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n\ttraining the mnodel (ie fine tuning)...")

    # Fine-tune the model
    trainer.train()

    print("\n\tevaluating the model...")

    # Step 7: Evaluate the model on the test dataset
    eval_results = trainer.evaluate()

    print("eval_results:\n", type(eval_results), eval_results)

    print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")

    return eval_results




    

if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Text Classification with LLAMA (testing).")
    
    parser.add_argument('--dataset', type=str, default='20newsgroups', help='Dataset to use: 20newsgroups or bbc-news.')

    parser.add_argument('--learner', type=str, default='svm', help='Choose the learner, in [nn, svm, fine].')
    
    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', help=f'dataset base vectorization strategy, in [tfidf, count]')                    
    
    parser.add_argument('--mix', type=str, default='solo', metavar='N', help=f'way to prepare the embeddings, in [vmode, solo, cat, dot]. NB presumes --pretrained is set')

    parser.add_argument('--optimc', action='store_true', help='Optimize classifier with GridSearchCV, only valid when using SVM learner.')
    
    parser.add_argument('--cm', action='store_true', help='Generate confusion matrix.')
    
    args = parser.parse_args()

    print("args:", args)
        
    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")

        # Number of GPUs available
        num_gpus = torch.cuda.device_count()
        print('Number of GPUs:', num_gpus)

        num_replicas = torch.cuda.device_count()
        print(f'Using {num_replicas} GPU(s)')
        
        # If using multiple GPUs, use DataParallel or DistributedDataParallel
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)    
        
    # Check for MPS availability (for Apple Silicon)
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")

        num_replicas = 1  # Use CPU or single GPU
        
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"          # disable memory upper limit

    # Default to CPU if neither CUDA nor MPS is available
    else:
        print("Neither CUDA nor MPS is available, using CPU")
        device = torch.device("cpu")
        
        num_replicas = 1  # Use CPU or single GPU
        

    print(f"Using device: {device}")

    dataset = args.dataset

    if (dataset not in ['20newsgroups', 'bbc_news', 'ohsumed']):
        print(f"Invalid dataset option. Please choose from '20newsgroups', 'bbc_news', or 'ohsumed'.")
        exit()

    print(f'loading dataset {dataset}...')
    
    X_train_raw, X_test_raw, y_train, y_test, target_names = load_dataset(dataset=dataset)

    print("X_train_raw:", type(X_train_raw), len(X_train_raw))
    print("X_test_raw:", type(X_test_raw), len(X_test_raw))
    print("y_train:", type(y_train), len(y_train))
    print("y_test:", type(y_test), len(y_test))
    print("target_names:", target_names)
    
    if (args.learner == 'svm'):

        # Load the pre-trained LLaMA model and tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrained=LLAMA_MODEL,
            max_length=TOKEN_TOKENIZER_MAX_LENGTH,
            cache_dir=VECTOR_CACHE + '/LLaMa',
            padding="max_length",
            truncation=True,
            return_tensors="pt"
            )
        
        """
        tokenizer = LlamaTokenizerFast.from_pretrained(
            LLAMA_MODEL, 
            max_length=TOKEN_TOKENIZER_MAX_LENGTH,
            cache_dir=VECTOR_CACHE + '/LLaMa'
            )
        """

        model = LlamaModel.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa').to(device)

        # Ensure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))  # Resize model's embeddings to accommodate the new token

        print("model:\n ", model)
        print("tokenizer:\n", tokenizer)

        # Custom tokenizer function using LLaMA tokenizer (no need for stopwords filtering here)
        def llama_tokenizer(text):
            tokens = tokenizer.tokenize(text)
            return tokens

        # Create a TF-IDF vectorizer with the custom tokenizer
        tfidf_vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, tokenizer=llama_tokenizer)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw).toarray()
        X_test_tfidf = tfidf_vectorizer.transform(X_test_raw).toarray()

        print("X_train_tfidf:", type(X_train_tfidf), X_train_tfidf.shape)
        print("X_train_tfidf[0]:", X_train_tfidf[0])
        print("X_test_tfidf:", type(X_test_tfidf), X_test_tfidf.shape)

        #print("X_train_tfidf[0]\n:", X_train_tfidf[0])

        llama_vocab_embeddings = get_llama_vocab_embeddings(
            embeddings_file=f'{PICKLE_DIR}/{dataset}_llama_vocab_embeddings_{MAX_VOCAB_SIZE}.pkl',
            tfidf_vectorizer=tfidf_vectorizer,
            tokenizer=tokenizer,
            device=device
            )
                
        print("llama_vocab_embeddings:", type(llama_vocab_embeddings), len(llama_vocab_embeddings))

        from itertools import islice

        print("llama_vocab_embeddings (first three elements):\n:")
        # Print the first 3 elements
        for key, value in islice(llama_vocab_embeddings.items(), 3):
            print(f'{key}, {value}\n')
           
        if (args.mix == 'solo'):

            print("\n\tApproach I: converting dataset into LlaMa embedding space (--solo)...")

            run_solo(
                X_train_tfidf=X_train_tfidf,
                X_test_tfidf=X_test_tfidf,
                y_train=y_train,
                y_test=y_test,
                target_names=target_names,
                vectorizer=tfidf_vectorizer,
                llama_vocab_embeddings=llama_vocab_embeddings,
                vocab=vocab
                )
        
        elif (args.mix == 'dot'):
        
            print("\n\tApproach II: projecting tfidf vectors into the LlaMa embedding space (vocabulary) using matrix multiplication (i.e. dot product)...")

            run_dot(
                X_train_tfidf=X_train_tfidf,
                X_test_tfidf=X_test_tfidf,
                y_train=y_train,
                y_test=y_test,
                target_names=target_names,
                vectorizer=tfidf_vectorizer,
                llama_vocab_embeddings=llama_vocab_embeddings,
                vocab=vocab
                )
        else:
            print(f'unsupported mix (--mix) {args,mix}, exiting...')
            exit(0)

    elif (args.learner == 'nn'):

        run_neural_model(
            args,
            device
            )
    
    elif (args.learner == 'fine'):
        
        fine_tune(
            X_train_raw,
            X_test_raw,
            y_train, y_test,
            num_labels=len(target_names),
    #        tokenizer=tokenizer,
            device=device
            )
    
    else:
        print(f'unsupported learner (--learner) {args.learner}, exiting...')
        exit(0)
        

