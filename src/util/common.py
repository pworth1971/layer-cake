import numpy as np
from tqdm import tqdm
import torch
import pandas as pd

# system and metrics reporting
import platform
import os
import platform
import psutil
import GPUtil

import matplotlib.pyplot as plt

import argparse

from scipy.sparse import issparse, csr_matrix
from joblib import Parallel, delayed

import multiprocessing
import itertools

from util.csv_log import CSVLog

from embedding.pretrained import GloVe, BERT, Word2Vec, FastText, LLaMA




NEURAL_MODELS = ['cnn', 'lstm', 'attn', 'ff']
ML_MODELS = ['svm', 'lr', 'nb']


SUPPORTED_LMS = ['glove', 'word2vec', 'fasttext', 'bert', 'roberta', 'xlnet', 'gpt2', 'llama']
SUPPORTED_TRANSFORMER_LMS = ['bert', 'roberta', 'xlnet', 'llama', 'gpt2']


OUT_DIR = '../out/'                                 # output directory

WORD_BASED_MODELS = ['glove', 'word2vec', 'fasttext']
TOKEN_BASED_MODELS = ['bert', 'roberta', 'gpt2', 'xlnet', 'llama']



def get_pretrained_embeddings(model, dataset):
    """
    returns Boolean and then data structure with embeddings, None if emb_model is not one if the acceptable values
    """

    print("loading pretrained embeddings for model type:", {model})

    return True, dataset.lcr_model, dataset.lcr_model.vocabulary
    
# ---------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------
# load_pretrained_embeddings()
#
# returns Boolean and then data structure with embeddings, None if 
# emb_model is not one if the acceptable values
#
# ---------------------------------------------------------------------------------------------------------------------------------------
def load_pretrained_embeddings(model, args):

    print("loading pretrained embeddings for model type:", {model})

    if model=='glove':
        print("path:", {args.glove_path})
        print("Loading GloVe pretrained embeddings...")
        glv = GloVe(path=args.glove_path)
        return True, glv, glv.vocabulary
    
    elif model=='word2vec':
        print("path:", {args.word2vec_path})
        print("Loading Word2Vec pretrained embeddings...")
        word2vec = Word2Vec(path=args.word2vec_path, limit=1000000)
        return True, word2vec, word2vec.vocabulary 
    
    elif model=='fasttext':
        print("path:", {args.fasttext_path})
        print("Loading fastText pretrained embeddings...")
        fasttext = FastText(path=args.fasttext_path, limit=1000000)
        return True, fasttext, fasttext.vocabulary
    
    """
    elif model=='bert':
        print("path:", {args.bert_path})
        print("Loading BERT pretrained embeddings...")
        return True, BERT(model_name=DEFAULT_BERT_PRETRAINED_MODEL, emb_path=args.bert_path)

    elif model=='llama':
        print("path:", {args.llama_path})
        print("Loading LLaMA pretrained embeddings...")
        return True, LLaMA(model_name=DEFAULT_LLAMA_PRETRAINED_MODEL, emb_path=args.llama_path)
    """

    return False, None
# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
def index(data, vocab, known_words, analyzer, unk_index, out_of_vocabulary):
    """
    Index (i.e., replaces word strings with numerical indexes) a list of string documents
    
    :param data: list of string documents
    :param vocab: a fixed mapping [str]->[int] of words to indexes
    :param known_words: a set of known words (e.g., words that, despite not being included in the vocab, can be retained
    because they are anyway contained in a pre-trained embedding set that we know in advance)
    :param analyzer: the preprocessor in charge of transforming the document string into a chain of string words
    :param unk_index: the index of the 'unknown token', i.e., a symbol that characterizes all words that we cannot keep
    :param out_of_vocabulary: an incremental mapping [str]->[int] of words to indexes that will index all those words that
    are not in the original vocab but that are in the known_words
    :return:
    """
    indexes=[]
    vocabsize = len(vocab)
    unk_count = 0
    knw_count = 0
    out_count = 0
    pbar = tqdm(data, desc=f'indexing documents')
    for text in pbar:
        words = analyzer(text)
        index = []
        for word in words:
            if word in vocab:
                idx = vocab[word]
            else:
                if word in known_words:
                    if word not in out_of_vocabulary:
                        out_of_vocabulary[word] = vocabsize+len(out_of_vocabulary)
                    idx = out_of_vocabulary[word]
                    out_count += 1
                else:
                    idx = unk_index
                    unk_count += 1
            index.append(idx)
        indexes.append(index)
        knw_count += len(index)
        pbar.set_description(f'[unk = {unk_count}/{knw_count}={(100.*unk_count/knw_count):.2f}%]'
                             f'[out = {out_count}/{knw_count}={(100.*out_count/knw_count):.2f}%]')
    return indexes
# ---------------------------------------------------------------------------------------------------------------------------------------


def define_pad_length(index_list):
    lengths = [len(index) for index in index_list]
    return int(np.mean(lengths)+np.std(lengths))


def pad(index_list, pad_index, max_pad_length=None):
    pad_length = np.max([len(index) for index in index_list])
    if max_pad_length is not None:
        pad_length = min(pad_length, max_pad_length)
    for i,indexes in enumerate(index_list):
        index_list[i] = [pad_index]*(pad_length-len(indexes)) + indexes[:pad_length]
    return index_list


def get_word_list(word2index1, word2index2=None): #TODO: redo
    print("get_word_list()")
    def extract_word_list(word2index):
        return [w for w,i in sorted(word2index.items(), key=lambda x: x[1])]
    word_list = extract_word_list(word2index1)
    if word2index2 is not None:
        word_list += extract_word_list(word2index2)
    return word_list


def batchify_old(index_list, labels, batchsize, pad_index, device, target_long=False, max_pad_length=500):
    
    print(f'batchify(): batchsize={batchsize}, pad_index={pad_index}, device={device}, target_long={target_long}, max_pad_length={max_pad_length}')
    
    #print("labels:", type(labels), labels.shape)
    #print("labels:", labels)

    nsamples = len(index_list)
    nbatches = nsamples // batchsize + 1*(nsamples%batchsize>0)
    
    #print(f'nsamples: {nsamples}, nbatches: {nbatches}')
    
    for b in range(nbatches):
        #print(f'processing batch {b}...')

        batch = index_list[b*batchsize:(b+1)*batchsize]
        batch_labels = labels[b*batchsize:(b+1)*batchsize]

        #print("batch_labels pre conversion:", type(batch_labels), batch_labels.shape)

        # Check batch_labels object type and convert as necessary
        if isinstance(batch_labels, pd.Series):
            batch_labels = batch_labels.astype(float)
        elif issparse(batch_labels):
            batch_labels = batch_labels.astype(float)
        
        #print("batch_labels post conversion:", type(batch_labels), batch_labels.shape)
        #print("batch_labels:", batch_labels)

        batch = pad(batch, pad_index=pad_index, max_pad_length=max_pad_length)
        batch = torch.LongTensor(batch)
        totype = torch.LongTensor if target_long else torch.FloatTensor
        target = totype(batch_labels)
        yield batch.to(device), target.to(device)


def batchify_unlabelled(index_list, batchsize, pad_index, device, max_pad_length=500):
    nsamples = len(index_list)
    nbatches = nsamples // batchsize + 1*(nsamples%batchsize>0)
    for b in range(nbatches):
        batch = index_list[b*batchsize:(b+1)*batchsize]
        batch = pad(batch, pad_index=pad_index, max_pad_length=max_pad_length)
        batch = torch.LongTensor(batch)
        yield batch.to(device)


def clip_gradient(model, clip_value=1e-1):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def predict(logits, classification_type='singlelabel'):
    if classification_type == 'multilabel':
        prediction = torch.sigmoid(logits) > 0.5
    elif classification_type == 'singlelabel':
        prediction = torch.argmax(logits, dim=1).view(-1, 1)
    else:
        print('unknown classification type')

    return prediction.detach().cpu().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parallel_slices(n_tasks, n_jobs=-1):
    if n_jobs==-1:
        n_jobs = multiprocessing.cpu_count()
    batch = int(n_tasks / n_jobs)
    remainder = n_tasks % n_jobs
    return [slice(job*batch, (job+1)*batch+ (remainder if job == n_jobs - 1 else 0)) for job in range(n_jobs)]


def tokenize_job(documents, tokenizer, max_tokens, job):
    return [tokenizer(d)[:max_tokens] for d in tqdm(documents, desc=f'tokenizing [job: {job}]')]


def tokenize_parallel(documents, tokenizer, max_tokens, n_jobs=-1):
    slices = get_parallel_slices(n_tasks=len(documents), n_jobs=n_jobs)
    tokens = Parallel(n_jobs=n_jobs)(
        delayed(tokenize_job)(
            documents[slice_i], tokenizer, max_tokens, job
        )
        for job, slice_i in enumerate(slices)
    )
    return list(itertools.chain.from_iterable(tokens))


def plot_loss_over_epochs(dataset, data, method_name, output_path):
    """
    Plots the training and testing loss over epochs.

    Parameters:
        data (dict): A dictionary containing 'epochs', 'train_loss', and 'test_loss'.
        method_name (str): The name of the method for labeling the plot.
        output_path (str): Path to save the plot.
    """
    epochs = data['epochs']
    train_loss = data['train_loss']
    test_loss = data['test_loss']

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, test_loss, label='Testing Loss', marker='x')
    plt.title(f'Loss Over for {method_name} - {dataset}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/{dataset}-{method_name}_loss.png')
    plt.close()


def print_arguments(options):
    print("Command Line Arguments:")
    for arg, value in vars(options).items():
        print(f"{arg}: {value}")


def get_sysinfo(debug=False):

    print("get_sysinfo()...")
    
    # Get CPU information
    num_physical_cores = psutil.cpu_count(logical=False)
    num_logical_cores = psutil.cpu_count(logical=True)

    print("CPU Physical Cores:", num_physical_cores)
    print("CPU Logical Cores:", num_logical_cores)
    #print("CPU Usage Per Core:", psutil.cpu_percent(percpu=True, interval=1))

    # Get memory information
    memory = psutil.virtual_memory()

    total_memory = memory.total
    avail_mem = memory.available

    print("Total Memory:", total_memory)
    print("Available Memory:", avail_mem)
    #print("Memory Usage %:", memory.percent)

    num_cuda_devices = 0
    cuda_devices = []                   # initialize cuda device list

    print("GPU Info...")
    if torch.cuda.is_available():
        num_cuda_devices = torch.cuda.device_count()
        print("Total CUDA devices:", num_cuda_devices)
        for i in range(num_cuda_devices):
            device_info = {
                "name": torch.cuda.get_device_name(i),
                "memory": torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)  # Convert bytes to MB
            }
            cuda_devices.append(device_info)
            #print(f"Device {i}: {device_info['name']} - Memory: {device_info['memory']} MB")
            print(f"Device {i}: {torch.cuda.get_device_name(i)} - Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)} MB")
    else:
        num_cuda_devices = 0
        print("CUDA is not available")

    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU: {gpu.name}, Memory Free: {gpu.memoryFree}MB, Memory Used: {gpu.memoryUsed}MB, Memory Total: {gpu.memoryTotal}MB")
    
    return num_physical_cores, num_logical_cores, total_memory, avail_mem, num_cuda_devices, cuda_devices


def get_model_computation_method(vtype='tfidf', pretrained=None, embedding_type='word', learner='???', mix=None):

    print("calculating model computation method...")

    print(f'vtype: {vtype}, pretrained: {pretrained}, embedding_type: {embedding_type}, learner: {learner}, mix: {mix}')

    if (learner in ML_MODELS):
        comp_method = 'ML'
    elif (learner in NEURAL_MODELS):
        comp_method = 'DL'
    else:
        comp_method = '??'

    if (pretrained is None):
        pt_type = 'pt=na'
    else:
        pt_type = 'pretrained-'

        if (pretrained in ['bert', 'roberta', 'llama', 'xlnet', 'distilbert', 'gpt2']):
            pt_type += 'attention:tokenized'

        elif (pretrained in ['glove', 'word2vec']):
            pt_type += 'co-occurrence:word'
        
        elif (pretrained in ['fasttext']):
            pt_type += 'co-occurrence:subword'
        else:
            pt_type += '??'
                
    if (learner in ML_MODELS): 

        if mix in ['solo', 'solo-wce']:
            type_type = f'{pt_type}[{mix}]'
            
        elif mix == 'vmode':
            type_type = f'frequency:{vtype}'

        elif mix in ['cat-doc', 'cat-wce', 'cat-doc-wce']:
            type_type = f'frequency:{vtype}+{pt_type}'

        elif mix in ['dot', 'dot-wce']:
            type_type = f'frequency:{vtype}.{pt_type}'

        elif mix in ['lsa', 'lsa-wce']:
            type_type = f'frequency:{vtype}->SVD'

        else:
            raise ValueError(f'Unknown mix type: {mix} for learner: {learner}')
       
        return comp_method + '-' + pt_type + '-' + type_type
    
    elif (learner in NEURAL_MODELS):
        return comp_method + '-' + pt_type


# ---------------------------------------------------------------------------------------------------------------------------

def index_dataset(dataset, tokenizer=None, max_length=512, pretrained=None):
    """
    Indexes the dataset using either a tokenizer (for Transformer models) or a pretrained word-based embedding model.
    """
    print(f"indexing dataset...")

    # For word-based models, use the dataset's vocabulary. For Transformer models, use the tokenizer's vocabulary.
    if tokenizer:
        word2index = dict(tokenizer.get_vocab())
        unk_index = tokenizer.unk_token_id
        pad_index = tokenizer.pad_token_id
    else:
        word2index = dict(dataset.vocabulary)
        word2index['UNKTOKEN'] = len(word2index)
        word2index['PADTOKEN'] = len(word2index)
        unk_index = word2index['UNKTOKEN']
        pad_index = word2index['PADTOKEN']

    known_words = set(word2index.keys())
    if pretrained is not None:
        known_words.update(pretrained.vocabulary())

    out_of_vocabulary = dict()

    # Define a helper function to tokenize and index documents
    def tokenize_and_index(documents):
        indices = []
        for doc in documents:
            if tokenizer:
                # Use the transformer tokenizer for subword tokenization
                tokens = tokenizer.encode(doc, truncation=True, padding=True, max_length=max_length)
            else:
                # Use the dataset analyzer for word-based tokenization
                tokens = dataset.analyzer()(doc)

            # Convert tokens to indices, handle OOVs
            indexed_tokens = [word2index.get(token, unk_index) for token in tokens]
            indices.append(indexed_tokens)
        return indices

    # Index development and test sets
    devel_index = tokenize_and_index(dataset.devel_raw)
    test_index = tokenize_and_index(dataset.test_raw)

    print('[indexing complete]')

    return word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index




def index_dataset_old(dataset, pretrained=None):

    print(f"indexing dataset...")
    
    # retreive the vocabulary
    word2index = dict(dataset.vocabulary)
    known_words = set(word2index.keys())
    if pretrained is not None:
        known_words.update(pretrained.vocabulary())

    word2index['UNKTOKEN'] = len(word2index)
    word2index['PADTOKEN'] = len(word2index)
    unk_index = word2index['UNKTOKEN']
    pad_index = word2index['PADTOKEN']

    # index documents and keep track of test terms outside the development vocabulary that are in GloVe (if available)
    out_of_vocabulary = dict()
    analyzer = dataset.analyzer()
    devel_index = index(dataset.devel_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary)
    test_index = index(dataset.test_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary)

    print('[indexing complete]')

    return word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index



# ---------------------------------------------------------------------------------------------------------------------------

def get_embeddings_path(pretrained, args):
    
    if (pretrained == 'bert'):
        return args.bert_path
    elif pretrained == 'roberta':
        return args.roberta_path
    elif pretrained == 'distilbert':
        return args.distilbert_path
    elif pretrained == 'glove':
        return args.glove_path
    elif pretrained == 'word2vec':
        return args.word2vec_path
    elif pretrained == 'fasttext':
        return args.fasttext_path
    elif pretrained == 'llama':
        return args.llama_path
    elif pretrained == 'xlnet':
        return args.xlnet_path
    elif pretrained == 'gpt2':
        return args.gpt2_path
    else:
        return args.glove_path          # default to GloVe embeddings
        


def get_language_model_type(embeddings):

    if (embeddings in ['glove']):
        return 'static:word:co-occurrence:global'
    elif (embeddings in ['word2vec']):
        return 'static:word:co-occurrence:local'
    elif (embeddings in ['fasttext']):
        return 'static:subword:co-occurrence:local'
    elif (embeddings in ['llama', 'gpt2']):
        return 'transformer:token:autoregressive:unidirectional:causal'
    elif (embeddings in ['bert', 'roberta']):
        return 'transformer:token:autoregressive:bidirectional:masked'
    elif (embeddings in ['distilbert']):
        return 'transformer:token:bidirectional:masked'
    elif (embeddings in ['xlnet']):
        return 'transformer:token:autoregressive:bidirectional:permutated'
    else:
        return 'static:word:co-occurrence:global'           # default to word embeddings
    
# ---------------------------------------------------------------------------------------------------------------------------


def get_embedding_type(pretrained):

    if (pretrained is not None and pretrained in ['bert', 'roberta', 'llama', 'xlnet', 'gpt2', 'llama']):
        embedding_type = 'token'
    elif (pretrained is not None and pretrained in ['glove', 'word2vec']):
        embedding_type = 'word'
    elif (pretrained is not None and pretrained in ['fasttext']):
        embedding_type = 'subword'
    else:
        embedding_type = 'word'             # default to word embeddings

    return embedding_type



def set_method_name(opt):
    
    method_name = opt.net
    
    if opt.pretrained:
        method_name += f'-{opt.pretrained}'
    
    if opt.learnable is not None and opt.learnable > 0:
        method_name += f'-learn{opt.learnable}'
    
    if opt.supervised:
        sup_drop = 0 if opt.droptype != 'sup' else opt.dropprob
        #method_name += f'-supervised-d{sup_drop}-{opt.supervised_method}'
        method_name += f'-wce-d{sup_drop}-{opt.supervised_method}'
    
    if opt.dropprob > 0:
        if opt.droptype != 'sup':
            method_name += f'-drop{opt.droptype}{opt.dropprob}'
    
    if (opt.pretrained or opt.supervised) and opt.tunable:
        method_name+='-tunable'
    elif (opt.pretrained or opt.supervised) and not opt.tunable:
        method_name+='-static'

    if opt.weight_decay > 0:
        method_name+=f'_wd{opt.weight_decay}'
    
    if opt.net in {'lstm', 'attn'}:
        method_name+=f'-h{opt.hidden}'
    
    if opt.net== 'cnn':
        method_name+=f'-ch{opt.channels}'
    
    return method_name


def initialize_testing(args):

    print("\n\tinitializing...")

    print("args:", args)

    # get system info to be used for logging below
    num_physical_cores, num_logical_cores, total_memory, avail_mem, num_cuda_devices, cuda_devices = get_sysinfo()

    cpus = f'physical:{num_physical_cores},logical:{num_logical_cores}'
    mem = total_memory
    if (num_cuda_devices >0):
        #gpus = f'{num_cuda_devices}:type:{cuda_devices[0]}'
        gpus = f'{num_cuda_devices}:{cuda_devices[0]}'
    else:
        gpus = 'None'

    method_name = set_method_name(args)
    print("method_name:", method_name)

    logger = CSVLog(
        file=args.log_file, 
        columns=[
            'os',
            'cpus',
            'mem',
            'gpus',
            'dataset', 
            'class_type',
            'model', 
            'embeddings',
            'lm_type',
            'mode',
            'comp_method',
            'representation',
            'optimized',
            'dimensions',
            'measure', 
            'value',
            'timelapse',
            'epoch',
            'run',
            ], 
        verbose=True, 
        overwrite=False)

    run_mode = method_name
    print("run_mode:", {run_mode})

    if args.pretrained:
        pretrained = True
    else:
        pretrained = False

    embeddings = args.pretrained
    print("embeddings:", {embeddings})

    lm_type = get_language_model_type(embeddings)
    print("lm_type:", {lm_type})

    if (args.supervised):
        supervised = True
        mode = 'supervised'
    else:
        supervised = False
        mode = 'unsupervised'

    # get the path to the embeddings
    emb_path = get_embeddings_path(embeddings, args)
    print("emb_path: ", {emb_path})

    system = SystemResources()
    print("system:\n", system)

    if (args.dataset in ['bbc-news', '20newsgroups']):
        logger.set_default('class_type', 'single-label')
    else:
        logger.set_default('class_type', 'multi-label')
        
    # set default system params
    logger.set_default('os', system.get_os())
    logger.set_default('cpus', system.get_cpu_details())
    logger.set_default('mem', system.get_total_mem())
    logger.set_default('mode', run_mode)

    gpus = system.get_gpu_summary()
    if gpus is None:
        gpus = -1   
    logger.set_default('gpus', gpus)

    logger.set_default('dataset', args.dataset)
    logger.set_default('model', args.net)
    logger.set_default('mode', mode)
    logger.set_default('embeddings', embeddings)
    logger.set_default('run', args.seed)
    logger.set_default('representation', method_name)
    logger.set_default('lm_type', lm_type)
    logger.set_default('optimized', args.tunable)

    embedding_type = get_embedding_type(embeddings)
    print("embedding_type:", embedding_type)

    comp_method = get_model_computation_method(
        vtype=args.vtype,
        pretrained=embeddings, 
        embedding_type=embedding_type, 
        learner=args.net, 
        mix=None
        )
    print("comp_method:", comp_method)
    logger.set_default('comp_method', comp_method)

    # check to see if the model has been run before
    already_modelled = logger.already_calculated(
        dataset=args.dataset,
        #embeddings=embeddings,
        model=args.net, 
        representation=method_name,
        mode=mode,
        run=args.seed
        )

    print("already_modelled:", already_modelled)

    return already_modelled, logger, method_name, pretrained, embeddings, emb_path, lm_type, mode, system


# ------------------------------------------------------------------------------------------------------------------------------------------
# parse_config_file()
# 
# Parses a config file of command line arguments for batch processing, used by both layer_cake and ml_baseline code
# 
def parse_config_file(config_file, parser):

    print("parse_config_file:", config_file)

    """
    # Create a default Namespace from parser
    args_defaults = argparse.Namespace(**{action.dest: action.default for action in parser._actions})

    print("args_defaults:", type(args_defaults), args_defaults)
    """

    # Create a default Namespace from parser
    # Collect default values and expected types
    args_defaults = argparse.Namespace()
    type_info = {}
    for action in parser._actions:
        setattr(args_defaults, action.dest, action.default)
        type_info[action.dest] = action.type if action.type else str

    print("args_defaults:", type(args_defaults), args_defaults)

    configurations = []

    with open(config_file, 'r') as file:

        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments

            #print("line:", line)
            # Make a deepcopy of the default arguments for each configuration
            current_args = copy.deepcopy(args_defaults)

            args_dict = {}
            for pair in line.split(","):  # Assuming each argument pair is separated by a comma
                key, value = map(str.strip, pair.split(':'))
                key = key.replace('-', '_')  # Normalize the key to match Namespace attribute

                # Handle boolean values
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif key in type_info:
                    # Convert value to the correct type
                    try:
                        value = type_info[key](value)
                    except ValueError:
                        raise ValueError(f"Error converting value for {key} on line {line_number}: expected type {type_info[key]}")

                args_dict[key] = value

            # Update the Namespace with arguments from the line
            for key, value in args_dict.items():
                setattr(current_args, key, value)

            # Store the fully updated Namespace for later use or directly call layer_cake
            configurations.append(current_args)

    return configurations        


# -------------------------------------------------------------------------------------------------------------------------------------------------

def todense(y):
    """Convert sparse matrix to dense format as needed."""
    return y.toarray() if issparse(y) else y


def tosparse(y):
    """Ensure matrix is in CSR format for efficient arithmetic operations."""
    return y if issparse(y) else csr_matrix(y)

# -------------------------------------------------------------------------------------------------------------------------------------------------





class SystemResources:
    def __init__(self):
        self.os = self.get_os()
        self.cpus = self.get_cpus()
        self.mem = self.get_mem()
        self.gpus = self.get_gpus()

        self.num_physical_cores, self.num_logical_cores, self.total_memory, self.avail_mem, self.num_cuda_devices, self.cuda_devices = get_sysinfo()

    def get_os(self):
        return platform.platform()

    def get_cpus(self):
        return os.cpu_count()

    def get_cpu_details(self):
        return f'physical:{self.num_physical_cores},logical:{self.num_logical_cores}'

    def get_mem(self):
        mem = psutil.virtual_memory()
        return {'total': mem.total, 'available': mem.available, 'percent': mem.percent}
        
    def get_total_mem(self):
        return self.total_memory

    def get_gpus(self):
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                'name': gpu.name,
                'total_memory': gpu.memoryTotal,
                'available_memory': gpu.memoryFree,
                'utilization': gpu.load
            })
        return gpu_info

    def get_gpu_summary(self):
        if (self.num_cuda_devices >0):
            #gpus = f'{num_cuda_devices}:type:{cuda_devices[0]}'
            return f'{self.num_cuda_devices}:{self.cuda_devices[0]}'
        else:
            return None

    def __str__(self):
        return f"OS: {self.os}, CPUs: {self.cpus}, Memory: {self.mem}, GPUs: {self.gpus}"




def get_system_resources():

    print("get_system_resources()...")

    operating_system = platform.platform()
    print("Operating System:", operating_system)

    # get system info to be used for logging below
    num_physical_cores, num_logical_cores, total_memory, avail_mem, num_cuda_devices, cuda_devices = get_sysinfo()

    cpus = f'physical:{num_physical_cores},logical:{num_logical_cores}'
    mem = total_memory

    gpus = 'None'
    if (num_cuda_devices >0):
        #gpus = f'{num_cuda_devices}:type:{cuda_devices[0]}'
        gpus = f'{num_cuda_devices}:{cuda_devices[0]}'

    return operating_system, cpus, mem, gpus





