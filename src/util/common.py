import numpy as np
from tqdm import tqdm
import os
import pickle
import torch

from scipy.sparse import vstack, issparse

from joblib import Parallel, delayed

import multiprocessing
import itertools
import platform
import psutil
import GPUtil


# custom importa
from util.csv_log import CSVLog

from embedding.pretrained import GLOVE_MODEL, WORD2VEC_MODEL, FASTTEXT_MODEL
from embedding.pretrained import BERT_MODEL, ROBERTA_MODEL, DISTILBERT_MODEL
from embedding.pretrained import XLNET_MODEL, GPT2_MODEL

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



# --------------------------------------------------------------------------------------------------------------
#
PICKLE_DIR = '../pickles/'                                          # pickle directory

OUT_DIR = '../out/'                                                 # output directory
LOG_DIR = '../log/'                                                 # log directory

VECTOR_CACHE = '../.vector_cache'                                   # vector cache directory (for language models)
DATASET_DIR = '../datasets/'                                        # dataset directory


NEURAL_MODELS = ['cnn', 'lstm', 'attn', 'ff', 'hf.sc.ff', 'hf.class.ff']
ML_MODELS = ['svm', 'lr', 'nb']

SUPPORTED_LMS = ['glove', 'word2vec', 'fasttext', 'bert', 'roberta', 'distilbert', 'xlnet', 'gpt2']
SUPPORTED_TRANSFORMER_LMS = ['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2']

WORD_BASED_MODELS = ['glove', 'word2vec', 'fasttext']
TOKEN_BASED_MODELS = ['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2']
#
# --------------------------------------------------------------------------------------------------------------


def initialize_testing(args, program, version):
    
    import datetime

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

    representation = set_method_name(args)
    print("representation:", representation)

    logger = CSVLog(
        file=args.log_file, 
        columns=[
            'source',
            'version',
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
            'timestamp'  # Added timestamp column
            ], 
        verbose=True, 
        overwrite=False)

    """
    run_mode = representation
    print("run_mode:", {run_mode})
    """

    if args.pretrained:
        pretrained = True
    else:
        pretrained = False

    embeddings = args.pretrained
    print("embeddings:", {embeddings})

    lm_type = get_language_model_type(embeddings)
    print("lm_type:", {lm_type})

    """
    if (args.supervised):
        supervised = True
        mode = 'supervised'
    else:
        supervised = False
        mode = 'unsupervised'
    """

    if (args.supervised):
        supervised = True
        mode = f'supervised[{args.sup_mode}]'

        if not args.nozscore:
            mode += '-zscore'
    else:
        supervised = False
        mode = f'unsupervised'

    # get the path to the embeddings
    emb_path = get_embeddings_path(embeddings, args)
    print("emb_path: ", {emb_path})

    system = SystemResources()
    print("system:\n", system)

    if (args.dataset in ['bbc-news', '20newsgroups', 'imdb']):
        logger.set_default('class_type', 'single-label')
    else:
        logger.set_default('class_type', 'multi-label')
        
    # set default system params
    logger.set_default('os', system.get_os())
    logger.set_default('cpus', system.get_cpu_details())
    logger.set_default('mem', system.get_total_mem())
    logger.set_default('mode', representation)

    logger.set_default('source', program)
    logger.set_default('version', version)
    
    gpus = system.get_gpu_summary()
    if gpus is None:
        gpus = -1   
    logger.set_default('gpus', gpus)

    logger.set_default('dataset', args.dataset)
    logger.set_default('model', args.net)
    logger.set_default('mode', mode)
    logger.set_default('embeddings', embeddings)
    logger.set_default('run', args.seed)
    logger.set_default('representation', representation)
    logger.set_default('lm_type', lm_type)
    logger.set_default('optimized', args.tunable)

    # Add the current timestamp
    current_timestamp = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    logger.set_default('timestamp', current_timestamp)

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
        model=args.net, 
        representation=representation,
        mode=mode,
        run=args.seed,
        optimized=args.tunable
        )

    print("already_modelled:", already_modelled)

    return already_modelled, logger, representation, pretrained, embeddings, embedding_type, emb_path, lm_type, mode, system



def initialize_testing_deprecated(args, program, version):

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
            'source',
            'version',
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

    """
    if (args.supervised):
        supervised = True
        mode = 'supervised'
    else:
        supervised = False
        mode = 'unsupervised'
    """

    if (args.supervised):
        supervised = True
        mode = f'supervised:{args.sup_mode}'

        if not args.nozscore:
            mode += '-zscore'
    else:
        supervised = False
        mode = f'unsupervised'

    # get the path to the embeddings
    emb_path = get_embeddings_path(embeddings, args)
    print("emb_path: ", {emb_path})

    system = SystemResources()
    print("system:\n", system)

    if (args.dataset in ['bbc-news', '20newsgroups', 'imdb']):
        logger.set_default('class_type', 'single-label')
    else:
        logger.set_default('class_type', 'multi-label')
        
    # set default system params
    logger.set_default('os', system.get_os())
    logger.set_default('cpus', system.get_cpu_details())
    logger.set_default('mem', system.get_total_mem())
    logger.set_default('mode', run_mode)

    logger.set_default('source', program)
    logger.set_default('version', version)

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

    return already_modelled, logger, method_name, pretrained, embeddings, embedding_type, emb_path, lm_type, mode, system



def get_embeddings_path(pretrained, args):
    
    if pretrained == 'glove':
        return args.glove_path
    elif pretrained == 'word2vec':
        return args.word2vec_path
    elif pretrained == 'fasttext':
        return args.fasttext_path
    elif (pretrained == 'bert'):
        return args.bert_path
    elif pretrained == 'roberta':
        return args.roberta_path
    elif pretrained == 'distilbert':
        return args.distilbert_path
    elif pretrained == 'xlnet':
        return args.xlnet_path
    elif pretrained == 'gpt2':
        return args.gpt2_path
    else:
        return args.glove_path          # default to GloVe embeddings
    
    """
    elif pretrained == 'albert':
        return args.albert_path
    elif pretrained == 'llama':
        return args.llama_path
    """

        

def get_language_model_type(embeddings, model_name=None):
    """
    Determines the type of language model based on the embedding type and model name.

    Parameters:
    ----------
    embeddings : str
        The type of embedding (e.g., 'glove', 'fasttext', etc.).
    model_name : str, optional
        The specific model name or file name of the embedding. Used to differentiate 
        between word-based and subword-based embeddings for FastText.

    Returns:
    -------
    str
        A descriptive string representing the language model type.
    """
    if embeddings in ['glove']:
        return 'static:word:co-occurrence:global'
    elif embeddings in ['word2vec']:
        return 'static:word:co-occurrence:local'
    elif embeddings in ['fasttext']:
        if model_name and "subword" in model_name.lower():
            return 'static:subword:co-occurrence:local'
        else:
            return 'static:word:co-occurrence:local'
    elif embeddings in ['bert', 'roberta']:
        return 'transformer:token:autoregressive:bidirectional:masked'
    elif embeddings in ['distilbert']:
        return 'transformer:token:bidirectional:masked'
    elif embeddings in ['albert']:
        return 'transformer:token:bidirectional:sop:masked'
    elif embeddings in ['xlnet']:
        return 'transformer:token:autoregressive:bidirectional:permutated'
    elif embeddings in ['llama', 'gpt2']:
        return 'transformer:token:autoregressive:unidirectional:causal'
    else:
        return 'static:word:co-occurrence:global'  # Default to word embeddings


def set_method_name(opt, add_model=False):
    
    method_name = opt.net
    
    if opt.pretrained:

        method_name += f'-{opt.pretrained}'

        if (add_model):

            # Add model type and specific model name for supported pretrained models
            pretrained_model_details = {
                'glove': ('static', GLOVE_MODEL),
                'word2vec': ('static', WORD2VEC_MODEL),
                'fasttext': ('subword', FASTTEXT_MODEL),
                'bert': ('transformer', BERT_MODEL),
                'roberta': ('transformer', ROBERTA_MODEL),
                'distilbert': ('transformer', DISTILBERT_MODEL),
                'xlnet': ('transformer', XLNET_MODEL),
                'gpt2': ('transformer', GPT2_MODEL),
            }
            
            # Extract the model type and specific model name
            model_details = pretrained_model_details.get(opt.pretrained.lower(), ('unknown', 'unknown'))
            model_type, model_name = model_details
            
            # Append to the method name
            method_name += f':{model_name}'
    
    if opt.learnable is not None and opt.learnable > 0:
        method_name += f'-learn{opt.learnable}'
    
    if opt.supervised:
        sup_drop = 0 if opt.droptype != 'sup' else opt.dropprob
        #method_name += f'-supervised-d{sup_drop}-{opt.supervised_method}'

        """
        if (opt.pretrained in ['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2']):
            method_name += f'-tce[{opt.sup_mode}]-d{sup_drop}-{opt.supervised_method}'
        else:
            method_name += f'-wce({opt.sup_mode})-d{sup_drop}-{opt.supervised_method}-{opt.pretrained}'
        """
        
        method_name += f'-wce({opt.sup_mode})-d{sup_drop}-{opt.supervised_method}-{opt.pretrained}'
        
        if not opt.nozscore:
            method_name += '-zscore'
        else:
            method_name += '-nozscore'

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



def get_embedding_type(pretrained):

    print("get_embedding_type():", {pretrained})

    if (pretrained is not None and pretrained in ['bert', 'roberta', 'distilbert', 'albert', 'xlnet', 'gpt2', 'llama']):
        embedding_type = 'token'
    elif (pretrained is not None and pretrained in ['glove', 'word2vec']):
        embedding_type = 'word'
    elif (pretrained is not None and pretrained in ['fasttext']):
        embedding_type = 'subword'
    else:
        embedding_type = 'word'             # default to word embeddings

    return embedding_type


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

        if (pretrained in ['bert', 'roberta', 'distilbert', 'albert', 'xlnet', 'gpt2', 'llama']):
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
        
    def get_num_gpus(self):
        return self.num_cuda_devices

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




# ------------------------------------------------------------------------------------------------------------------------------------------------




def index_dataset(dataset, opt, pt_model=None):
    """
    Indexes the dataset for use with word-based (e.g., GloVe, Word2Vec, FastText)
    and token-based (e.g., BERT, RoBERTa, DistilBERT) embeddings.

    Parameters:
    ----------
    dataset : Dataset
        The dataset object containing raw text and a fitted vectorizer.
    opt : argparse.Namespace
    pt_model : PretrainedEmbeddings class (instantiated) optional
        Pretrained embedding object to extend the known vocabulary.

    Returns:
    -------
    tuple : (word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index)
        - word2index: Mapping of tokens to indices.
        - out_of_vocabulary: Mapping of OOV tokens to indices.
        - unk_index: Index for unknown tokens.
        - pad_index: Index for padding tokens.
        - devel_index: Indexed representation of the development dataset.
        - test_index: Indexed representation of the test dataset.
    """
    print(f'indexing dataset.... dataset: {dataset}, vtype: {opt.vtype}, model: {opt.pretrained}')
    
    pickle_file = os.path.join(PICKLE_DIR, f"{opt.dataset}.{opt.vtype}.{opt.pretrained}.index.pickle")
    print('pickle_file:', pickle_file)

    # Check if the pickle file exists
    if os.path.exists(pickle_file):
        print(f"Loading indexed dataset from {pickle_file}...")
        with open(pickle_file, "rb") as f:
            return pickle.load(f)

    print("Indexing dataset from scratch...")

    # Build the vocabulary
    word2index = dict(dataset.vocabulary)
    known_words = set(word2index.keys())
    if pt_model is not None:
        print(f"Updating known_words with pretrained vocabulary: {len(pt_model.vocabulary())} entries")
        known_words.update(pt_model.vocabulary())
    print(f"Total known words: {len(known_words)}")

    # Add special tokens
    word2index['UNKTOKEN'] = len(word2index)
    word2index['PADTOKEN'] = len(word2index)
    unk_index = word2index['UNKTOKEN']
    pad_index = word2index['PADTOKEN']

    # Initialize out-of-vocabulary dictionary and analyzer
    out_of_vocabulary = dict()
    analyzer = dataset.analyzer()
    print("Analyzer initialized.")

    # Index the datasets
    devel_index = index(dataset.devel_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary, opt)
    test_index = index(dataset.test_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary, opt)

    # Save the indexed dataset to a pickle file
    print(f"Saving indexed dataset to {pickle_file}...")
    with open(pickle_file, "wb") as f:
        pickle.dump((word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index), f)

    return word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index





def index_dataset_old(dataset, opt, pt_model=None):
    """
    Indexes the dataset for use with word-based (e.g., GloVe, Word2Vec, FastText)
    and token-based (e.g., BERT, RoBERTa, DistilBERT) embeddings.

    Parameters:
    ----------
    dataset : Dataset
        The dataset object containing raw text and a fitted vectorizer.
    opt : argparse.Namespace
    pt_model : PretrainedEmbeddings class (instantiated) optional
        Pretrained embedding object to extend the known vocabulary.

    Returns:
    -------
    tuple : (word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index)
        - word2index: Mapping of tokens to indices.
        - out_of_vocabulary: Mapping of OOV tokens to indices.
        - unk_index: Index for unknown tokens.
        - pad_index: Index for padding tokens.
        - devel_index: Indexed representation of the development dataset.
        - test_index: Indexed representation of the test dataset.
    """
    print('indexing dataset...')

    # build the vocabulary
    word2index = dict(dataset.vocabulary)
    known_words = set(word2index.keys())
    if pt_model is not None:
        print(f'updating known_words with pretrained.vocabulary(): {type(pt_model.vocabulary())}; {len(pt_model.vocabulary())}')
        known_words.update(pt_model.vocabulary())
    print("known_words:", type(known_words), len(known_words))

    word2index['UNKTOKEN'] = len(word2index)
    word2index['PADTOKEN'] = len(word2index)
    unk_index = word2index['UNKTOKEN']
    pad_index = word2index['PADTOKEN']

    # index documents and keep track of test terms outside the 
    # development vocabulary that are in pretrained model (if available)
    out_of_vocabulary = dict()
    analyzer = dataset.analyzer()
    print("analyzer:", type(analyzer), analyzer)

    devel_index = index(dataset.devel_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary, opt)
    test_index = index(dataset.test_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary, opt)

    return word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index



def index(data, vocab, known_words, analyzer, unk_index, out_of_vocabulary, opt):
    """
    Index (i.e., replaces word strings with numerical indexes) a list of string documents and log outputs to files.

    :param data: list of string documents
    :param vocab: a fixed mapping [str]->[int] of words to indexes
    :param known_words: a set of known words (e.g., words that are not in vocab but exist in the pre-trained embeddings)
    :param analyzer: the preprocessor in charge of transforming the document string into a chain of string words
    :param unk_index: the index of the 'unknown token', representing all words not in vocab or known_words
    :param out_of_vocabulary: an incremental mapping [str]->[int] of words to indexes that are not in vocab but in known_words
    :param opt: the options object containing the dataset, vector type, and pre-trained embeddings

    :return: indexed documents
    """
    
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    file_prefix = f"{opt.dataset}_{opt.vtype}.{opt.pretrained}"

    # Define output file paths
    text_and_tokens_file = os.path.join(OUT_DIR, f"{file_prefix}_text_tokens.txt")
    vocab_words_file = os.path.join(OUT_DIR, f"{file_prefix}_vocab_words.txt")
    known_words_file = os.path.join(OUT_DIR, f"{file_prefix}_known_words.txt")
    unknown_words_file = os.path.join(OUT_DIR, f"{file_prefix}_unknown_words.txt")

    # Initialize output containers
    indexed_docs = []
    vocabsize = len(vocab)
    unk_count = 0
    knw_count = 0
    out_count = 0

    # Prepare sets for unique words
    all_vocab_words = set()
    all_known_words = set()
    all_unknown_words = set()

    # Process each document
    with open(text_and_tokens_file, 'w') as tt_file:
        pbar = tqdm(data, desc=f'indexing documents')
        for text in pbar:
            words = analyzer(text)
            index = []
            tt_file.write(f"Text: {text}\nTokens: {words}\n\n")
            for word in words:
                if word in vocab:
                    idx = vocab[word]
                    all_vocab_words.add(word)
                else:
                    if word in known_words:
                        if word not in out_of_vocabulary:
                            out_of_vocabulary[word] = vocabsize + len(out_of_vocabulary)
                        idx = out_of_vocabulary[word]
                        out_count += 1
                        all_known_words.add(word)
                    else:
                        idx = unk_index
                        unk_count += 1
                        all_unknown_words.add(word)
                index.append(idx)
            indexed_docs.append(index)
            knw_count += len(index)
            pbar.set_description(f'[unk = {unk_count}/{knw_count}={(100.*unk_count/knw_count):.2f}%]'
                                 f'[out = {out_count}/{knw_count}={(100.*out_count/knw_count):.2f}%]')


    # Write vocab words, known words, and unknown words to files
    with open(vocab_words_file, 'w') as vocab_file:
        vocab_file.write("\n".join(sorted(all_vocab_words)))

    with open(known_words_file, 'w') as known_file:
        known_file.write("\n".join(sorted(all_known_words)))

    with open(unknown_words_file, 'w') as unknown_file:
        unknown_file.write("\n".join(sorted(all_unknown_words)))

    """
    print(f"Text and tokens written to: {text_and_tokens_file}")
    print(f"Vocabulary words written to: {vocab_words_file}")
    print(f"Known words written to: {known_words_file}")
    print(f"Unknown words written to: {unknown_words_file}")
    """

    return indexed_docs




def index_old(data, vocab, known_words, analyzer, unk_index, out_of_vocabulary):
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
    def extract_word_list(word2index):
        return [w for w,i in sorted(word2index.items(), key=lambda x: x[1])]
    word_list = extract_word_list(word2index1)
    if word2index2 is not None:
        word_list += extract_word_list(word2index2)
    return word_list


def batchify(index_list, labels, batchsize, pad_index, device, target_long=False, max_pad_length=500):
    nsamples = len(index_list)
    nbatches = nsamples // batchsize + 1*(nsamples%batchsize>0)
    for b in range(nbatches):
        batch = index_list[b*batchsize:(b+1)*batchsize]
        batch_labels = labels[b*batchsize:(b+1)*batchsize]
        if issparse(batch_labels):
            batch_labels = batch_labels.toarray()
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