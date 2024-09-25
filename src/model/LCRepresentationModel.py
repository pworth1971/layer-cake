import numpy as np
import os
import torch
from tqdm import tqdm

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.language_representation.representation_model import batch_iterable, mean_across_all_tokens, \
    concat_all_tokens

from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import BertTokenizerFast, LlamaTokenizerFast, RobertaTokenizerFast
from transformers import BertModel, LlamaModel, RobertaModel


from abc import ABC, abstractmethod

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from gensim.models.fasttext import load_facebook_model

from util.common import VECTOR_CACHE, DATASET_DIR


# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 8
MPS_BATCH_SIZE = 16


# Setup device prioritizing CUDA, then MPS, then CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    BATCH_SIZE = DEFAULT_GPU_BATCH_SIZE
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    BATCH_SIZE = MPS_BATCH_SIZE
else:
    DEVICE = torch.device("cpu")
    BATCH_SIZE = DEFAULT_CPU_BATCH_SIZE



# -------------------------------------------------------------------------------------------------------
# default pretrained models.
#
# NB: these models are all case sensitive, ie no need to lowercase the input text (see _preprocess)
#
#GLOVE_MODEL = 'glove.6B.300d.txt'                          # dimension 300, case insensensitve
GLOVE_MODEL = 'glove.42B.300d.txt'                          # dimensiomn 300, case sensitive

WORD2VEC_MODEL = 'GoogleNews-vectors-negative300.bin'       # dimension 300, case sensitive

FASTTEXT_MODEL = 'crawl-300d-2M-subword.bin'                # dimension 300, case sensitive

#BERT_MODEL = 'bert-base-cased'                              # dimension = 768, case sensitive
BERT_MODEL = 'bert-large-cased'                             # dimension = 1024, case sensitive

#ROBERTA_MODEL = 'roberta-base'                             # dimension = 768, case insensitive
ROBERTA_MODEL = 'roberta-large'

LLAMA_MODEL = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096, case sensitive
#LLAMA_MODEL = 'meta-llama/Llama-2-13b-hf'
# -------------------------------------------------------------------------------------------------------


MIN_DF_COUNT = 5
MAX_VOCAB_SIZE = 15000                                      # max feature size for TF-IDF vectorization


#
# tokens for LLAMA model access, must be requested from huggingface
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'






class LCRepresentationModel(RepresentationModel, ABC):
    """
    class LCRepresentationModel(RepresentationModel): 
    inherits from the transformer RepresentationModel class and acts as an abstract base class which
    computes representations of dataset text for different pretrained language models. This version 
    implements an encoding method that returns a summary vector for the different models as well, with 
    some performance optimizations.
    """
    
    def __init__(self, model_name=None, model_dir='../.vector_cache'):
        """
        Initialize the representation model.
        
        Args:
        - model_type (str): Type of the model ('bert', 'roberta', etc.).
        - model_name (str): Hugging Face Model Hub ID or local path.
        - model_dir (str, optional): Directory to save and load model files.
        - device (str, optional): Device to use for encoding ('cuda', 'mps', or 'cpu').
        """
        
        print("initializing LCRepresentationModel...")

        self.device = DEVICE
        self.batch_size = BATCH_SIZE
        print("self.device:", self.device)
        print("self.batch_size:", self.batch_size)

        self.model_name = model_name
        print("self.model_name:", self.model_name)
        
        self.model_dir = model_dir
        print("self.model_dir:", model_dir)

        self.combine_strategy = 'mean'          # default combine strategy  

        self.model = None
        self.tokenizer = None

        self.initialized = True

    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    

    @abstractmethod
    def encode_docs(self, text_list, embedding_vocab_matrix):
        """
        Abstract method to be implemented by all subclasses.
        """
        pass




class WordLCRepresentationModel(LCRepresentationModel):
    """
    WordBasedRepresentationModel handles word-based embeddings (e.g., Word2Vec, GloVe).
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name, model_dir, vtype='tfidf', model_type='word2vec'):
        """
        Initialize the word-based representation model (e.g., Word2Vec, GloVe).
        
        Parameters:
        ----------
        model_name : str
            Name of the pre-trained word embedding model (e.g., 'word2vec', 'glove').
        embedding_path : str
            Path to the pre-trained embedding file (e.g., 'GoogleNews-vectors-negative300.bin').
        device : str, optional
            Device to use for encoding ('cpu' for word embeddings since it's lightweight).
        """
        print("Initializing WordBasedRepresentationModel...")

        super().__init__(model_name, model_dir=model_dir)  # parent constructor

        path_to_embeddings = model_dir + '/' + model_name
        print("path_to_embeddings:", path_to_embeddings)

        if (model_type == 'word2vec'):
            print("Using Word2Vec pretrained embeddings...")
            self.model = KeyedVectors.load_word2vec_format(path_to_embeddings, binary=True)
        elif (model_type == 'glove'):
            print("Using GloVe pretrained embeddings...")
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove_input_file = path_to_embeddings
            word2vec_output_file = glove_input_file + '.word2vec'
            glove2word2vec(glove_input_file, word2vec_output_file)
            self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        else:
            raise ValueError("Invalid model type. Use 'word2vec' or 'glove'.")
        
        self.vtype = vtype
        print(f"Vectorization type: {vtype}")

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.vector_size
        print(f"Embedding dimension: {self.embedding_dim}")

        #
        # vectorize the text, note that the Word2Vec and GloVe models we use are case sensitive
        #
        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                sublinear_tf=True,                              # use sublinear TF scaling
                lowercase=False                                 # dont lowercase the tokens
            )              
        elif vtype == 'count':
            print("using Count vectorization...")

            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                lowercase=False                                 # dont lowercase the tokens
            )
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")
        


    def build_embedding_vocab_matrix(self):
        
        print("building [word based] embedding representation (matrix) of dataset vocabulary...")

        print("model:", type(self.model))
        print("model.vector_size:", self.model.vector_size)

        self.embedding_dim = self.model.vector_size
        self.vocab_size = len(self.vectorizer.vocabulary_)
        print("embedding_dim:", self.embedding_dim)
        print("vocab_size:", self.vocab_size)

        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        # Dictionary to store the index-token mapping
        self.token_to_index_mapping = {}

        oov_tokens = 0

        # Loop through the dataset vocabulary and fill the embedding matrix
        for word, idx in self.vectorizer.vocabulary_.items():
            self.token_to_index_mapping[idx] = word  # Store the token at its index

            # For Word2Vec and GloVe (retrieving word vectors)
            if word in self.model.key_to_index:  # Check if the word exists in the pretrained model
                self.embedding_vocab_matrix[idx] = self.model[word]
            else:
                #print(f"Warning: OOV word when building embedding vocab matrix. Word: '{word}'")
                # Handle OOV words with a random embedding (you could also use a zero vector if preferred)
                self.embedding_vocab_matrix[idx] = np.random.normal(size=(self.embedding_dim,))
                oov_tokens += 1

        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print("token_to_index_mapping:", type(self.token_to_index_mapping), len(self.token_to_index_mapping))
        print("oov_tokens:", oov_tokens)

        return self.embedding_vocab_matrix, self.token_to_index_mapping




    def encode_docs(self, texts, embedding_vocab_matrix):
        """
        Compute both weighted document embeddings (using TF-IDF) and average document embeddings for each document.
        
        Args:
        - texts: List of input documents (as raw text).
        - embedding_vocab_matrix: Matrix of pre-trained word embeddings for Word2Vec/GloVe/FastText.

        Returns:
        - weighted_document_embeddings: Numpy array of weighted document embeddings for each document.
        - avg_document_embeddings: Numpy array of average document embeddings for each document.
        """
        
        print(f"encoding docs...")
        
        print("texts:", type(texts), len(texts))
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

        weighted_document_embeddings = []
        avg_document_embeddings = []

        unk_token_id = self.vectorizer.vocabulary_.get('<unk>', None)  # Check for the existence of an UNK token
        print("unk_token_id:", unk_token_id)
        
        oov_tokens = 0

        for doc in texts:

            # Tokenize the document using the vectorizer (ensures consistency in tokenization)
            tokens = self.vectorizer.build_analyzer()(doc)

            # Calculate TF-IDF weights for the tokens
            tfidf_vector = self.vectorizer.transform([doc]).toarray()[0]

            weighted_sum = np.zeros(embedding_vocab_matrix.shape[1])
            total_weight = 0.0
            valid_embeddings = []    

            for token in tokens:
                #token_lower = token.lower()

                #token_id = self.vectorizer.vocabulary_.get(token.lower(), None)
                token_id = self.vectorizer.vocabulary_.get(token, None)
                #print("token:", token, "token_id:", token_id)

                if token_id is not None and 0 <= token_id < embedding_vocab_matrix.shape[0]:
                    
                    embedding = embedding_vocab_matrix[token_id]            # Get the embedding for the token
                    weight = tfidf_vector[token_id]                         # Get the TF-IDF weight for the token

                    # Accumulate the weighted embedding
                    weighted_sum += embedding * weight
                    total_weight += weight    

                elif unk_token_id is not None:
                    embedding = embedding_vocab_matrix[unk_token_id]        # Fallback to <unk> embedding if available
                else:
                    # OOV token handling: Use a zero vector if no embedding is found
                    #print(f"Warning: OOV token: {token}")
                    embedding = np.zeros(embedding_vocab_matrix.shape[1])
                    oov_tokens += 1
                
                valid_embeddings.append(embedding)
                
            # Compute the weighted embedding for the document
            if total_weight > 0:
                weighted_doc_embedding = weighted_sum / total_weight
            else:
                weighted_doc_embedding = np.zeros(embedding_vocab_matrix.shape[1])  # Handle empty or OOV cases

            # Compute the average embedding for the document
            if valid_embeddings:
                avg_doc_embedding = np.mean(valid_embeddings, axis=0)
            else:
                avg_doc_embedding = np.zeros(embedding_vocab_matrix.shape[1])  # Handle empty or OOV cases

            weighted_document_embeddings.append(weighted_doc_embedding)
            avg_document_embeddings.append(avg_doc_embedding)

        print("weighted_document_embeddings:", type(weighted_document_embeddings), len(weighted_document_embeddings))
        print("avg_document_embeddings:", type(avg_document_embeddings), len(avg_document_embeddings))

        print("oov_tokens:", oov_tokens)

        return np.array(weighted_document_embeddings), np.array(avg_document_embeddings)
    




class SubWordLCRepresentationModel(LCRepresentationModel):
    """
    SubWordBasedRepresentationModel handles subword-based embeddings (e.g. fastText).
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name=FASTTEXT_MODEL, model_dir=VECTOR_CACHE+'/fastText', vtype='tfidf'):
        """
        Initialize the word-based representation model (e.g., Word2Vec, GloVe).
        
        Parameters:
        ----------
        model_name : str
            Name of the pre-trained word embedding model (e.g. 'crawl-300d-2M-subword.bin').
        model_dir : str
            Path to the pre-trained embedding file (e.g., '../.vector_cache/fastText').
        vtype : str, optional
            vectorization type, either 'tfidf' or 'count'.
        """

        print("Initializing SubWordBasedRepresentationModel...")

        super().__init__(model_name, model_dir=model_dir)  # parent constructor

        # Load the pre-trained word embedding model
        # Append the FastText model name to the pretrained_path
        fasttext_model_path = model_dir + '/' + model_name
        print("fasteext_model_path:", fasttext_model_path)

        # Use load_facebook_model to load FastText model from Facebook's pre-trained binary files
        self.model = load_facebook_model(fasttext_model_path)
            
        self.vtype = vtype
        print(f"Vectorization type: {vtype}")

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.vector_size
        print(f"Embedding dimension: {self.embedding_dim}")

        #
        # vectorize the text, note that the Word2Vec and GloVe models we use are case sensitive
        #
        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                sublinear_tf=True,                              # use sublinear TF scaling
                lowercase=False                                 # dont lowercase the tokens
            )              
        elif vtype == 'count':
            print("using Count vectorization...")

            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                lowercase=False                                 # dont lowercase the tokens
            )
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")


    def build_embedding_vocab_matrix(self):

        print("building [subword based] embedding representation (matrix) of dataset vocabulary...")

        print("model:", type(self.model))
        print("model.vector_size:", self.model.vector_size)

        self.embedding_dim = self.model.vector_size
        self.vocab_size = len(self.vectorizer.vocabulary_)
        print("embedding_dim:", self.embedding_dim)
        print("vocab_size:", self.vocab_size)

        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        # Dictionary to store the index-token mapping
        self.token_to_index_mapping = {}

        oov_tokens = 0
        
        #
        # create the embedding matrix that aligns with the vocabulary
        #
        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        # Loop through the dataset vocabulary
        for word, idx in self.vectorizer.vocabulary_.items():
            self.token_to_index_mapping[idx] = word  # Store the token at its index

            if word in self.model.wv.key_to_index:  # If the word exists in FastText's vocabulary
                self.embedding_vocab_matrix[idx] = self.model.wv[word]
            else:
                #print(f"Warning: OOV word when building embedding vocab matrix. Word: '{word}'")
                # FastText automatically handles subword embeddings for OOV words
                self.embedding_vocab_matrix[idx] = self.model.wv.get_vector(word)
                oov_tokens += 1

        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print("token_to_index_mapping:", type(self.token_to_index_mapping), len(self.token_to_index_mapping))
        print("oov_tokens:", oov_tokens)

        return self.embedding_vocab_matrix, self.token_to_index_mapping


    def encode_docs(self, texts, embedding_vocab_matrix):
        """
        Compute both weighted document embeddings (using TF-IDF) and average document embeddings for each document.
        
        Args:
        - texts: List of input documents (as raw text).
        - embedding_vocab_matrix: Matrix of pre-trained word embeddings for Word2Vec/GloVe/FastText.

        Returns:
        - weighted_document_embeddings: Numpy array of weighted document embeddings for each document.
        - avg_document_embeddings: Numpy array of average document embeddings for each document.
        """
        
        print(f"encoding docs..")
        
        print("texts:", type(texts), len(texts))
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

        weighted_document_embeddings = []
        avg_document_embeddings = []

        oov_tokens = 0

        for doc in texts:
            
            # Tokenize the document using the vectorizer (ensures consistency in tokenization)
            tokens = self.vectorizer.build_analyzer()(doc)

            # Calculate TF-IDF weights for the tokens
            tfidf_vector = self.vectorizer.transform([doc]).toarray()[0]

            weighted_sum = np.zeros(embedding_vocab_matrix.shape[1])
            total_weight = 0.0
            valid_embeddings = []

            for token in tokens:
                # Get the embedding from FastText
                try:
                    embedding = self.model.wv.get_vector(token)  # FastText handles subword embeddings here
                except KeyError:
                    # If the token or subwords cannot be found (rare with FastText), use a zero vector
                    #print(f"Warning: OOV token: {token}")
                    embedding = np.zeros(embedding_vocab_matrix.shape[1])
                    oov_tokens += 1

            # Get the TF-IDF weight if available, else assign a default weight of 1 (or 0 if not in vocabulary)
            weight = tfidf_vector[self.vectorizer.vocabulary_.get(token, 0)]

            # Compute the weighted embedding for the document
            if total_weight > 0:
                weighted_doc_embedding = weighted_sum / total_weight
            else:
                weighted_doc_embedding = np.zeros(embedding_vocab_matrix.shape[1])

            # Compute the average embedding for the document
            if valid_embeddings:
                avg_doc_embedding = np.mean(valid_embeddings, axis=0)
            else:
                avg_doc_embedding = np.zeros(embedding_vocab_matrix.shape[1])  # Handle empty or OOV cases

            # Append the document embeddings to the list
            weighted_document_embeddings.append(weighted_doc_embedding)
            avg_document_embeddings.append(avg_doc_embedding)

        print("weighted_document_embeddings:", type(weighted_document_embeddings), len(weighted_document_embeddings))
        print("avg_document_embeddings:", type(avg_document_embeddings), len(avg_document_embeddings))

        print("oov_tokens:", oov_tokens)

        return np.array(weighted_document_embeddings), np.array(avg_document_embeddings)






class BERTRootLCRepresentationModel(LCRepresentationModel):
    """
    Shared class for BERT and RoBERTa models to implement common functionality, such as tokenization.
    """
    
    def __init__(self, model_name, model_dir, vtype='tfidf'):
        super().__init__(model_name, model_dir)
        
        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT, 
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")
    
    
    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the tokenizer, returning tokenized strings (not token IDs) for TF-IDF or CountVectorizer.
        This tokenizer works for BERT, RoBERTa, and LLaMA models.
        
        Parameters:
        - text: The input text to be tokenized.
        
        Returns:
        - tokens: A list of tokens with special tokens removed based on the model in use.
        """

        tokens = self.tokenizer.tokenize(text, max_length=self.max_length, truncation=True)
        
        # Retrieve special tokens from the tokenizer object
        special_tokens = self.tokenizer.all_special_tokens  # Dynamically fetch special tokens like [CLS], [SEP], <s>, </s>, etc.
        
        # Optionally, remove special tokens
        tokens = [token for token in tokens if token not in special_tokens]

        return tokens
    
    

    def _tokenize(self, texts):
        """
        Tokenize a batch of texts using the tokenizer, returning token IDs and attention masks.
        This method works for both BERT and RoBERTa models.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',    # Return PyTorch tensors
            padding=True,           # Pad the sequences to the longest sequence in the batch
            truncation=True,        # Truncate sequences that exceed the max length
            max_length=self.max_length  # Use the tokenizer's max length
        )
        
        return inputs['input_ids'], inputs['attention_mask']
    

    def build_embedding_vocab_matrix(self):
        """
        Build the embedding vocabulary matrix for BERT and RoBERTa models.
        """
        print("building [BERT/RoBERTa based] embedding representation (matrix) of dataset vocabulary...")

        self.model.eval()  # Set model to evaluation mode            
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")  # To confirm which device is being used
        print("model:", type(self.model))
        
        #
        # NB we use different embedding vocab matrices here depending upon the pretrained model
        #
        self.embedding_dim = self.model.config.hidden_size  # Get the embedding dimension size
        self.vocab_size = len(self.vectorizer.vocabulary_)

        print("embedding_dim:", self.embedding_dim)
        print("dataset vocab size:", self.vocab_size)   

        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        # Dictionary to store the index-token mapping
        self.token_to_index_mapping = {}

        # -------------------------------------------------------------------------------------------------------------
        def _bert_embedding_vocab_matrix(vocab, batch_size=DEFAULT_CPU_BATCH_SIZE, max_len=512):
            """
            Construct the embedding vocabulary matrix for BERT and RoBERTa models.
            """

            print("_bert_embedding_vocab_matrix...")

            print("batch_size:", batch_size)
            print("max_len:", max_len)

            embedding_vocab_matrix = np.zeros((len(vocab), self.model.config.hidden_size))
            batch_words = []
            word_indices = []
            
            def process_batch(batch_words, max_len):

                inputs = self.tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
                
                # move input tensors to proper device
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)  # Ensure all inputs are on the same device
                max_vocab_size = self.model.config.vocab_size

                # check for out of range tokens
                out_of_range_ids = input_ids[input_ids >= max_vocab_size]
                if len(out_of_range_ids) > 0:
                    print("Warning: The following input IDs are out of range for the model's vocabulary:")
                    for out_id in out_of_range_ids.unique():
                        token = self.tokenizer.decode(out_id.item())
                        print(f"Token '{token}' has ID {out_id.item()} which is out of range (vocab size: {max_vocab_size}).")
                
                # Perform inference, ensuring that the model is on the same device as the inputs
                self.model.to(self.device)
                with torch.no_grad():
                    #outputs = model(**inputs)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Return the embeddings and ensure they are on the CPU for further processing
                return outputs.last_hidden_state[:, 0,:].cpu().numpy()


            with tqdm(total=len(vocab), desc="processing bert / roberta embedding vocab matrix construction batches") as pbar:
                for word, idx in vocab.items():
                    batch_words.append(word)
                    word_indices.append(idx)

                    if len(batch_words) == batch_size:
                        embeddings = process_batch(batch_words, max_len)
                        for i, embedding in zip(word_indices, embeddings):
                            if i < len(embedding_vocab_matrix):
                                embedding_vocab_matrix[i] = embedding
                            else:
                                print(f"IndexError: Skipping index {i} as it's out of bounds for embedding_vocab_matrix.")
                                oov_tokens += 1
                        
                        batch_words = []
                        word_indices = []
                        pbar.update(batch_size)

                if batch_words:
                    embeddings = process_batch(batch_words, max_len)
                    for i, embedding in zip(word_indices, embeddings):
                        if i < len(embedding_vocab_matrix):
                            embedding_vocab_matrix[i] = embedding
                        else:
                            print(f"IndexError: Skipping index {i} as it's out of bounds for the embedding_vocab_matrix.")
                            oov_tokens += 1
                    
                    pbar.update(len(batch_words))

            return embedding_vocab_matrix

        #tokenize and prepare inputs
        max_length = self.tokenizer.model_max_length
        #print("max_length:", max_length)

        self.embedding_vocab_matrix = _bert_embedding_vocab_matrix(self.vocab, batch_size=BATCH_SIZE, max_len=max_length)
        
        # Add to token-to-index mapping for BERT and RoBERTa
        for word, idx in self.vectorizer.vocabulary_.items():
            self.token_to_index_mapping[idx] = word  # Store the token at its index

        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print("token_to_index_mapping:", type(self.token_to_index_mapping), len(self.token_to_index_mapping))

        return self.embedding_vocab_matrix, self.token_to_index_mapping




class BERTLCRepresentationModel(BERTRootLCRepresentationModel):
    """
    BERTRepresentation subclass implementing sentence encoding using BERT
    """
    
    def __init__(self, model_name=BERT_MODEL, model_dir=VECTOR_CACHE+'/BERT', vtype='tfidf'):

        print("initializing BERT representation model...")

        super().__init__(model_name, model_dir)                             # parent constructor

        # instantiate model and tokenizer
        self.model = BertModel.from_pretrained(model_name, cache_dir=model_dir)
        
        self.tokenizer = BertTokenizerFast.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            do_lower_case=False                 # keep tokenizer case sensitive
        )

        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        # NB BertTokenizerFast has a pad_token
            
        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT, 
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")
        
        self.model.to(self.device)      # put the model on the appropriate device
    

    def encode_docs(self, text_list, embedding_vocab_matrix=None):
        """
        Generates both the mean and first token embeddings for a list of text sentences using BERT.
        
        This function computes:
        - The mean of all token embeddings.
        - The first token embedding (position 0, [CLS] token for BERT).

        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.

        Returns:
        -------
        mean_embeddings : np.ndarray
            Array of mean sentence embeddings (mean of all tokens).
        first_token_embeddings : np.ndarray
            Array of sentence embeddings using the [CLS] token.
        """

        print("encoding docs using BERT...")

        self.model.eval()
        mean_embeddings = []
        first_token_embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]  # Token-level embeddings from RoBERTa

                # Compute the mean of all token embeddings
                batch_mean_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                mean_embeddings.append(batch_mean_embeddings)

                # Compute the first token embedding (similar to [CLS] in BERT)
                batch_first_token_embeddings = token_vectors[:, 0, :].cpu().detach().numpy()
                first_token_embeddings.append(batch_first_token_embeddings)

        # Concatenate all batch results
        mean_embeddings = np.concatenate(mean_embeddings, axis=0)
        first_token_embeddings = np.concatenate(first_token_embeddings, axis=0)

        return mean_embeddings, first_token_embeddings



class RoBERTaLCRepresentationModel(BERTRootLCRepresentationModel):
    """
    RoBERTaRepresentationModel subclass implementing sentence encoding using RoBERTa
    """

    def __init__(self, model_name=ROBERTA_MODEL, model_dir=VECTOR_CACHE+'/RoBERTa', vtype='tfidf'):

        print("initializing RoBERTa representation model...")

        super().__init__(model_name, model_dir)                             # parent constructor

        self.model = RobertaModel.from_pretrained(model_name, cache_dir=model_dir)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name, cache_dir=model_dir)
    
        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        # NB RoBERTaTokenizerFast has a pad_token

        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT, 
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")
        
        self.model.to(self.device)      # put the model on the appropriate device
    

    def encode_docs(self, text_list, embedding_vocab_matrix=None):
        """
        Generates both the mean and first token embeddings for a list of text sentences using RoBERTa.
        
        RoBERTa does not use a [CLS] token, but the first token often serves a similar purpose.
        This function computes:
        - The mean of all token embeddings.
        - The first token embedding (position 0).

        Parameters:
        ----------
        text_list : list of str
            List of docs to encode.

        Returns:
        -------
        mean_embeddings : np.ndarray
            Array of mean sentence embeddings (mean of all tokens).
        first_token_embeddings : np.ndarray
            Array of sentence embeddings using the first token.
        """

        print("encoding docs using RoBERTa...")

        self.model.eval()
        mean_embeddings = []
        first_token_embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]  # Token-level embeddings from RoBERTa

                # Compute the mean of all token embeddings
                batch_mean_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                mean_embeddings.append(batch_mean_embeddings)

                # Compute the first token embedding (similar to [CLS] in BERT)
                batch_first_token_embeddings = token_vectors[:, 0, :].cpu().detach().numpy()
                first_token_embeddings.append(batch_first_token_embeddings)

        # Concatenate all batch results
        mean_embeddings = np.concatenate(mean_embeddings, axis=0)
        first_token_embeddings = np.concatenate(first_token_embeddings, axis=0)


        return mean_embeddings, first_token_embeddings

    


class LlaMaLCRepresentationModel(LCRepresentationModel):
    """
    LLAMARepresentation subclass implementing sentence encoding using LLAMA
    """
    
    def __init__(self, model_name=LLAMA_MODEL, model_dir=VECTOR_CACHE+'/LlaMa', vtype='tfidf'):
        
        print("initializing LLAMA representation model...")

        super().__init__(model_name, model_dir)                             # parent constructor
    
        self.model = LlamaModel.from_pretrained(model_name, cache_dir=model_dir)
        
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            model_name, 
            cache_dir=model_dir
        )

        # Ensure padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding
            
        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT, 
                lowercase=False, 
                sublinear_tf=True, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")          

        self.model.to(self.device)      # put the model on the appropriate device


    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the tokenizer, returning tokenized strings (not token IDs) for TF-IDF or CountVectorizer.
        This tokenizer works for BERT, RoBERTa, and LLaMA models.
        
        Parameters:
        - text: The input text to be tokenized.
        
        Returns:
        - tokens: A list of tokens with special tokens removed based on the model in use.
        """

        #tokens = self.tokenizer.tokenize(text, max_length=self.max_length, truncation=True)

        tokens = self.tokenizer.tokenize(text, truncation=True)         # should pick up the max_length based upon model

        # Retrieve special tokens from the tokenizer object
        special_tokens = self.tokenizer.all_special_tokens  # Dynamically fetch special tokens like [CLS], [SEP], <s>, </s>, etc.
        
        # Optionally, remove special tokens
        tokens = [token for token in tokens if token not in special_tokens]

        return tokens
    

    def _tokenize(self, texts):
        """
        Tokenize a batch of texts using the LLaMa tokenizer, returning token IDs and attention masks.

        Parameters:
        ----------
        texts : list of str
            List of sentences to tokenize.

        Returns:
        -------
        input_ids : torch.Tensor
            Tensor of token IDs for each sentence.
        attention_mask : torch.Tensor
            Tensor of attention masks (1 for real tokens, 0 for padding tokens).
        """

        # Tokenize the input texts, padding/truncating them to the max length and returning tensors
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',            # Return PyTorch tensors
            padding=True,                   # Pad the sequences to the longest sequence in the batch
            truncation=True                # Truncate sequences that exceed the max length
            #max_length=self.max_length           # Define a max length (usually 512 for LLaMa)
        )
        
        # Return input IDs and attention mask as tensors
        return inputs['input_ids'], inputs['attention_mask']


    def build_embedding_vocab_matrix(self):

        print("building [LlaMa based] embedding representation (matrix) of dataset vocabulary...")

        self.model.eval()  # Set model to evaluation mode            
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")  # To confirm which device is being used
        print("model:", type(self.model))
        
        #
        # NB we use different embedding vocab matrices here depending upon the pretrained model
        #
        self.embedding_dim = self.model.config.hidden_size  # Get the embedding dimension size
        self.vocab_size = len(self.vectorizer.vocabulary_)

        print("embedding_dim:", self.embedding_dim)
        print("dataset vocab size:", self.vocab_size)   

        # Dictionary to store the index-token mapping
        self.token_to_index_mapping = {}

        # Collect all tokens from the vectorizer's vocabulary
        tokens = list(self.vectorizer.vocabulary_.keys())
        num_tokens = len(tokens)
        print("Number of tokens to encode:", num_tokens)

        # Batch size for processing
        batch_size = BATCH_SIZE

        # Initialize an empty NumPy array to store the embeddings
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float32)

        # Process tokens in batches
        for batch_start in tqdm(range(0, num_tokens, batch_size), desc="encoding dataset with LLaMa embeddings..."):
            batch_tokens = tokens[batch_start: batch_start + batch_size]

            # Tokenize the batch of tokens using the LLaMA tokenizer
            input_ids = self.tokenizer(
                batch_tokens,
                return_tensors='pt',
                padding=True,
                truncation=True,
                #max_length=max_length                  # should pick up max length from the underlying model
            ).input_ids.to(self.device)

            # Get the embeddings from the LLaMA model
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            # Mean pooling to get sentence-level embeddings for each token
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            # Store each token's embedding in the embedding matrix (direct index assignment)
            for i, token in enumerate(batch_tokens):
                # Find the index of the token in the vectorizer's vocabulary
                token_index = self.vectorizer.vocabulary_[token]
                self.embedding_matrix[token_index] = batch_embeddings[i]
                self.token_to_index_mapping[token_index] = token  # Store the token at its index

        print("embedding_matrix:", type(self.embedding_matrix), self.embedding_matrix.shape)
        print("token_to_index_mapping:", type(self.token_to_index_mapping), len(self.token_to_index_mapping))

        return self.embedding_matrix, self.token_to_index_mapping       



    def encode_docs(self, text_list, embedding_vocab_matrix=None):
        """
        Generates both the mean and last token embeddings for a list of text sentences using LLaMa.
        
        LLaMa does not use [CLS] tokens. Instead, it can return:
        - The mean of all token embeddings.
        - The last token embedding, which is a (representation of the) summary of the sentence.
        
        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.

        Returns:
        -------
        mean_embeddings : np.ndarray
            Array of mean sentence embeddings (mean of all tokens).
        last_embeddings : np.ndarray
            Array of sentence embeddings using the last token.
        """
        
        print("encoding docs using LLaMa...")

        self.model.eval()
        mean_embeddings = []
        last_embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]  # Token-level embeddings from LLaMa

                # Compute the mean of all token embeddings
                batch_mean_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                mean_embeddings.append(batch_mean_embeddings)

                # Compute the last token embedding
                batch_last_embeddings = token_vectors[:, -1, :].cpu().detach().numpy()
                last_embeddings.append(batch_last_embeddings)

        # Concatenate all batch results
        mean_embeddings = np.concatenate(mean_embeddings, axis=0)
        last_embeddings = np.concatenate(last_embeddings, axis=0)

        return mean_embeddings, last_embeddings

    











