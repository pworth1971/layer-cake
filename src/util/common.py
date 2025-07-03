import numpy as np
from tqdm import tqdm
import os
import pickle
import torch
import datetime
    
import pandas as pd
from nltk.corpus import stopwords
import string
import re


from scipy.sparse import vstack, issparse

from joblib import Parallel, delayed

import multiprocessing
import itertools
import platform
import psutil
import GPUtil

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


# custom importa
from util.csv_log import CSVLog

from model.LCRepresentationModel import GLOVE_MODEL, WORD2VEC_MODEL, FASTTEXT_MODEL, HYPERBOLIC_MODEL
from model.LCRepresentationModel import BERT_MODEL, ROBERTA_MODEL, DISTILBERT_MODEL
from model.LCRepresentationModel import XLNET_MODEL, GPT2_MODEL, DEEPSEEK_MODEL, LLAMA_MODEL
from model.LCRepresentationModel import MODEL_MAP, MODEL_DIR, VECTOR_CACHE, PICKLE_DIR

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --------------------------------------------------------------------------------------------------------------
#
OUT_DIR = '../out/'                                                 # output directory
LOG_DIR = '../log/'                                                 # log directory
DATASET_DIR = '../datasets/'                                        # dataset directory

NEURAL_MODELS = ['cnn', 'lstm', 'attn', 'hf.sc.ff', 'hf.sc', 'hf.cnn', 'linear']
ML_MODELS = ['svm', 'lr', 'nb']

SUPPORTED_LMS = ['glove', 'word2vec', 'fasttext', 'hyperbolic', 'bert', 'roberta', 'distilbert', 'xlnet', 'gpt2', 'llama', 'deepseek']
SUPPORTED_TRANSFORMER_LMS = ['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2', 'llama', 'deepseek']

WORD_BASED_MODELS = ['glove', 'word2vec', 'fasttext', 'hyperbolic']
TOKEN_BASED_MODELS = ['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2', 'llama', 'deepseek']
CLASS_EMBEDDING_MODES = ['solo-wce', 'dot-wce', 'cat-wce', 'lsa-wce', 'cat-doc-wce']
#
# --------------------------------------------------------------------------------------------------------------


def initialize_testing(args, model_name, program, version):
    """
    Initializes testing based on the provided arguments, used for neural models
    
    Args: 
        - args: A namespace or dictionary of arguments that specify the configuration for the test.
        Expected fields:
            - net (str): Type of learner to use ('svm', 'lr', or 'nb').
            - vtype (str): Vectorization type, either 'count' or 'tfidf'.
            - pretrained (str): Pretrained model or embedding type (e.g., 'BERT', 'LLaMA').
            - dataset (str): Name of the dataset.
            - log_file (str): Path to the log file where results will be stored.
            - mix (str): Dataset and embedding comparison method.
            - dataset_emb_comp (str): Dataset embedding comparison method.
        - model_name (str): Name of the model architecture.
        - program (str): Name of the program.
        - version (str): Version of the program.
        
    Returns:
        - already_modelled (bool): Whether the current configuration has already been modeled.
        - logger (CSVLog): Logger object to store model run details.
        - representation (str): Type of data representation used for training.
        - pretrained (bool): Whether to use a pretrained model or embeddings.
        - embeddings (str): Type of embeddings to use.
        - embedding_type (str): Type of embeddings, e.g., 'word', 'subword', 'token'.
        - emb_path (str): Path to the embeddings or pretrained model files.
        - lm_type (str): Language model type.
        - mode (str): Mode of operation, either 'supervised' or 'unsupervised'.
        - system (SystemResources): System resource information.
    """
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

    representation = get_representation(args)
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
            'classifier',
            'embeddings',
            'model', 
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
    if pretrained:
        emb_path = get_embeddings_path(embeddings, args)
        print("emb_path: ", {emb_path})
    else:
        emb_path = None
        print("emb_path: None")

    system = SystemResources()
    print("system:\n", system)

    if (args.dataset in ['bbc-news', '20newsgroups', 'imdb', 'arxiv_protoformer']):
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
    logger.set_default('classifier', args.net)
    logger.set_default('model', model_name)
    logger.set_default('mode', mode)
    logger.set_default('embeddings', embeddings)
    logger.set_default('run', args.seed)
    logger.set_default('representation', representation)
    logger.set_default('lm_type', lm_type)

    #
    # set optimized value (args.tunable) explcitly
    #
    optimized_val = 'none'
    if args.tunable:
        optimized_val = 'tunable'
    print("optimized_val:", optimized_val)
    logger.set_default('optimized', optimized_val)

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
        classifier=args.net, 
        representation=representation,
        mode=mode,
        run=args.seed,
        optimized=optimized_val,
        model=model_name
        )

    print("already_modelled:", already_modelled)

    return already_modelled, logger, representation, pretrained, embeddings, embedding_type, emb_path, lm_type, mode, system



# -------------------------------------------------------------------------------------------------------------------------------------------------
def initialize_ml_testing(args, model_name, program, version):
    """
    Initializes machine learning testing based on the provided arguments.

    Args:
        - args: A namespace or dictionary of arguments that specify the configuration for the ML experiment.
        Expected fields:
            - net (str): Type of learner to use ('svm', 'lr', or 'nb').
            - vtype (str): Vectorization type, either 'count' or 'tfidf'.
            - pretrained (str): Pretrained model or embedding type (e.g., 'BERT', 'LLaMA').
            - dataset (str): Name of the dataset.
            - logfile (str): Path to the log file where results will be stored.
            - mix (str): Dataset and embedding comparison method.
            - dataset_emb_comp (str): Dataset embedding comparison method.
        - model_name (str): Name of the model architecture.
        - program (str): Name of the program.
        - version (str): Version of the program.

    Returns:
        - already_computed (bool): Whether the current configuration has already been computed.
        - vtype (str): Vectorization type ('count' or 'tfidf').
        - learner (class): The ML model class to be used (e.g., `LinearSVC` for SVM).
        - pretrained (bool): Whether to use a pretrained model or embeddings.
        - embeddings (str): Type of embeddings to use.
        - emb_path (str): Path to the embeddings or pretrained model files.
        - mix (str): The dataset and embedding comparison method.
        - representation (str): Type of data representation used for training.
        - ml_logger (CSVLog): Logger object to store model run details.
        - optimized (bool): Whether the model is optimized for performance.
    """

    print(f"\n\tinitializing ML testing...  model_name: {model_name}, program: {program}, version: {version}")

    print("args:", args)

    # set up model type
    if args.net == 'svm':
        learner = LinearSVC
        learner_name = 'SVM' 
    elif args.net == 'lr':
        learner = LogisticRegression
        learner_name = 'LR'
    elif args.net == 'nb':
        learner = MultinomialNB
        #learner = GaussianNB
        learner_name = 'NB'
    else:
        print("** Unknown learner, possible values are svm, lr or nb **")
        return

    print("learner:", learner)
    print("learner_name: ", {learner_name})

    # default to tfidf vectorization type unless 'count' specified explicitly
    if args.vtype == 'count':
        vtype = 'count'
    else:
        vtype = 'tfidf'             
    print("vtype:", {vtype})

    if args.pretrained:
        pretrained = True
    else:
        pretrained = False
        
    if args.supervised:
        supervised = True
    else:
        supervised = False

    embeddings = args.pretrained
    print("embeddings:", {embeddings})

    lm_type = get_language_model_type(embeddings)
    print("lm_type:", {lm_type})

    # get the path to the embeddings
    emb_path = get_embeddings_path(args.pretrained, args)
    print("emb_path: ", {emb_path})

    model_type = f'{learner_name}:{args.vtype}-{args.mix}'
    print("model_type:", {model_type})
    
    print("initializing baseline layered log file...")

    logger = CSVLog(
        file=args.logfile, 
        columns=[
            'source',
            'version',
            'os',
            'cpus',
            'mem',
            'gpus',
            'dataset', 
            'class_type',
            'classifier', 
            'embeddings',
            'model',
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

    print("setting defaults...")

    print("pretrained: ", {pretrained}, "; embeddings: ", {embeddings})

    if (args.mix in [CLASS_EMBEDDING_MODES]):
        supervised = True
        mode = f'supervised'
        
        if not args.nozscore:
            mode += 'zscore'         # always use zscore with ML-WCE models
    else:
        supervised = False
        mode = f'unsupervised'

    mode += f'-[{args.vtype}.{args.mix}.{args.dataset_emb_comp}]'

    # get the path to the embeddings
    if pretrained:
        emb_path = get_embeddings_path(embeddings, args)
        print("emb_path: ", {emb_path})
    else:
        emb_path = None
        print("emb_path: None")
    
    system = SystemResources()
    print("system:\n", system)

    run_mode = args.dataset + ':' + args.net + ':' + args.pretrained + ':' + args.mix + ':' + args.dataset_emb_comp
    print("run_mode:", {run_mode})

    representation, optimized = get_ml_representation(args)
    print("representation:", {representation})

    # set default run time params
    logger.set_default('os', system.get_os())
    logger.set_default('cpus', system.get_cpu_details())
    logger.set_default('mem', system.get_total_mem())
    #logger.set_default('mode', run_mode)

    logger.set_default('source', program)
    logger.set_default('version', version)

    gpus = system.get_gpu_summary()
    if gpus is None:
        gpus = -1   
    logger.set_default('gpus', gpus)

    logger.set_default('dataset', args.dataset)
    logger.set_default('classifier', args.net)
    logger.set_default('model', model_name)
    logger.set_default('mode', mode)
    logger.set_default('embeddings', embeddings)
    logger.set_default('run', args.seed)
    logger.set_default('representation', representation)
    logger.set_default('lm_type', lm_type)

    #
    # set optimized value (args.tunable) explcitly
    #
    optimized_val = args.optimc
    print("optimized_val:", optimized_val)
    logger.set_default('optimized', optimized_val)

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
        mix=args.mix
        )
    print("comp_method:", comp_method)
    logger.set_default('comp_method', comp_method)

    # epoch and run fields are deprecated
    logger.set_default('epoch', -1)
    logger.set_default('run', -1)

    embeddings = get_embeddings_ml(args)
    print("embeddings:", {embeddings})

    lm_type = get_language_model_type(args.pretrained)
    print("lm_type:", {lm_type})

    # check to see if the model has been run before
    already_computed = logger.already_calculated(
        dataset=args.dataset,
        classifier=args.net,
        representation=representation,
        model=model_name,
        mode=mode,
        #embeddings=embeddings
        )

    print("already_computed:", already_computed)

    return already_computed, vtype, learner, pretrained, embeddings, lm_type, emb_path, args.mix, representation, logger, optimized

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def get_embeddings_ml(args):
    """
    build the embeddings field for ML logger
    """
    print("get_embeddings_ml...")

    if args.pretrained:
        if args.mix == 'vmode':
            return args.pretrained + ':' + args.mix + '.' + args.vtype
        else:
            return args.pretrained + ':' + args.mix
    else:
        return args.mix

    

def get_ml_representation(args):
    """
    set representation form
    """
    print("forming ML model representation...")

    # set model and dataset
    method_name = f'[{args.net}:{args.dataset}]:->'

    # vmode is when we simply use the frequency vector representation (TF-IDF or Count)
    # as the dataset representation into the model
    if (args.mix == 'vmode'):
        method_name += f'{args.vtype}[{args.pretrained}]'

    # solo is when we project the doc, we represent it, in the 
    # underlying pretrained embedding space - with three options 
    # as to how that space is computed: 1) weighted, 2) avg, 3) summary
    elif (args.mix == 'solo'):
        method_name += f'{args.pretrained}({args.dataset_emb_comp})'

    elif (args.mix == 'solo-wce'):
        method_name += f'{args.pretrained}({args.dataset_emb_comp})+{args.vtype}.({args.pretrained}+wce({args.vtype}))'

    # cat is when we concatenate the doc representation in the
    # underlying pretrained embedding space with the tfidf vectors - 
    # we have the same three options for the dataset embedding representation
    elif (args.mix == 'cat-doc'):
        method_name += f'{args.vtype}+{args.pretrained}({args.dataset_emb_comp})'

    elif (args.mix == 'cat-wce'):
        method_name += f'{args.vtype}+{args.vtype}.({args.pretrained}+wce({args.vtype}))'
    
    elif (args.mix == 'cat-doc-wce'):
        method_name += f'{args.vtype}+{args.vtype}.({args.pretrained}+wce({args.vtype}))+{args.vtype}.({args.pretrained}+wce({args.vtype}))'
        
    # dot is when we project the tfidf vectors into the underlying
    # pretrained embedding space using matrix multiplication, i.e. dot product
    # we have the same three options for the dataset embedding representation computation
    elif (args.mix == 'dot'):
        method_name += f'{args.vtype}.{args.pretrained}'

    elif (args.mix == 'dot-wce'):
        method_name += f'{args.vtype}.({args.pretrained}+wce({args.vtype}))'
        
    # lsa is when we use SVD (aka LSA) to reduce the number of featrues from 
    # the vectorized data set, LSA is a form of dimensionality reduction
    elif (args.mix == 'lsa'):
        method_name += f'{args.vtype}->LSA[{args.pretrained}].({args.pretrained})'

    elif (args.mix == 'lsa-wce'):
        method_name += f'{args.vtype}->LSA[{args.pretrained}]+({args.vtype}.{args.pretrained})+wce'

    #
    # set optimized field to true if its a neural model 
    # and we are tuning (fine-tuning) it or if its an ML
    # model and we are optimizing the prameters ofr best results
    #
    if (args.net in NEURAL_MODELS and args.tunable) or (args.net in ML_MODELS and args.optimc):
        method_name += ':[opt]'
        optimized = True
    else:
        method_name += ':[def]'
        optimized = False
    
    print("method_name:", method_name)

    return method_name, optimized


# Get the full model identifier and load from local directory
def get_model_identifier(pretrained, cache_dir=VECTOR_CACHE):
    
    print(f"get_model_identifier()... pretrained: {pretrained}, cache_dir: {cache_dir}")

    if pretrained is None:
        return None, None
    else:
        model_name = MODEL_MAP.get(pretrained, pretrained)        
        model_dir = MODEL_DIR.get(pretrained, pretrained)
        model_path = os.path.join(cache_dir, model_dir)
        return model_name, model_path
    

def get_embeddings_path(pretrained, args):
    
    print(f"get_embeddings_path()... pretrained: {pretrained}")
    
    if pretrained == 'glove':
        return args.glove_path
    elif pretrained == 'word2vec':
        return args.word2vec_path
    elif pretrained == 'fasttext':
        return args.fasttext_path
    elif pretrained == 'hyperbolic':
        return args.hyperbolic_path
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
    elif pretrained == 'llama':
        return args.llama_path
    elif pretrained == 'deepseek':
        return args.deepseek_path
    else:
        return args.glove_path          # default to GloVe embeddings

        

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
    elif embeddings in ['hyperbolic']:
        return 'static:word:hyperbolic'
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
    elif embeddings in ['deepseek']:
        return 'transformer:token:autoregressive:bidirectional:causal'
    else:
        return 'NA'  # Default to word embeddings



def get_representation(opt, add_model=False):
    
    print("getting model representation...")

    #
    # core model options
    #
    method_name = opt.net               # set model type
    
    if opt.learnable is not None and opt.learnable > 0:
        method_name += f'-learn{opt.learnable}'

    if opt.dropprob > 0:
        if opt.droptype != 'sup':
            method_name += f'-drop{opt.droptype}{opt.dropprob}'
        
    if opt.weight_decay > 0:
        method_name+=f'_wd{opt.weight_decay}'
    
    if opt.net in {'lstm', 'attn'}:
        method_name+=f'-h{opt.hidden}'
    
    if opt.net in {'cnn', 'hf.cnn'}:
        method_name+=f'-ch{opt.channels}'

    if opt.net == 'linear':
        method_name += f'-drop:{opt.dropprob}-pooling:{opt.pooling}'
    
    #
    # add pretrained parameters
    #    
    if opt.pretrained:

        method_name += f'-{opt.pretrained}'

        if (add_model):

            # Add model type and specific model name for supported pretrained models
            pretrained_model_details = {
                'glove': ('static', GLOVE_MODEL),
                'word2vec': ('static', WORD2VEC_MODEL),
                'fasttext': ('subword', FASTTEXT_MODEL),
                'hyperbolic': ('word', HYPERBOLIC_MODEL),
                'bert': ('transformer', BERT_MODEL),
                'roberta': ('transformer', ROBERTA_MODEL),
                'distilbert': ('transformer', DISTILBERT_MODEL),
                'xlnet': ('transformer', XLNET_MODEL),
                'gpt2': ('transformer', GPT2_MODEL),
                'llama': ('transformer', LLAMA_MODEL),
                'deepseek': ('transformer', DEEPSEEK_MODEL),
            }
            
            # Extract the model type and specific model name
            model_details = pretrained_model_details.get(opt.pretrained.lower(), ('unknown', 'unknown'))
            model_type, model_name = model_details
            
            # Append to the method name
            method_name += f':{model_name}'
    
        if opt.tunable:
            method_name += '.tunable'
        else:
            method_name += '.static'
    
    #
    # supervised options
    #
    if opt.supervised:
        
        sup_drop = 0 if opt.droptype != 'sup' else opt.dropprob

        #method_name += f'-supervised-d{sup_drop}-{opt.supervised_method}'

        if (opt.pretrained in ['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2', 'llama', 'deepseek']):

            method_name += '-tce'

            if opt.tunable_tces:
                method_name += '.tunable'

            if opt.normalize_tces:
                method_name += f'.norm[{opt.sup_mode}]'
            else:
                method_name += f'[{opt.sup_mode}]'    
        
        else:
            method_name += f'-wce[{opt.sup_mode}]'

        method_name += f'-d{sup_drop}-{opt.supervised_method}'

        if not opt.nozscore:
            method_name += '.zscore'
    

    return method_name



def get_embedding_type(pretrained):

    print("get_embedding_type():", {pretrained})

    if (pretrained is not None and pretrained in ['bert', 'roberta', 'distilbert', 'albert', 'xlnet', 'gpt2', 'llama', 'deepseek']):
        embedding_type = 'token'
    elif (pretrained is not None and pretrained in ['glove', 'word2vec', 'hyperbolic']):
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

    # Get memory information
    memory = psutil.virtual_memory()
    total_memory = memory.total
    avail_mem = memory.available

    print("Total Memory:", total_memory)
    print("Available Memory:", avail_mem)

    # Initialize CUDA info
    num_cuda_devices = 0
    cuda_devices = []

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
            print(f"Device {i}: {device_info['name']} - Memory: {device_info['memory']} MB")
    else:
        print("CUDA is not available")

    # GPUtil (typically for NVIDIA GPUs)
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU: {gpu.name}, Memory Free: {gpu.memoryFree}MB, Memory Used: {gpu.memoryUsed}MB, Memory Total: {gpu.memoryTotal}MB")

    # MPS / Metal on macOS
    if torch.backends.mps.is_available():
        print("MPS is available.")
        print("MPS GPU Info:")

        try:
            output = subprocess.check_output(
                ['system_profiler', '-xml', 'SPDisplaysDataType', '-detailLevel', 'full']
            )
            plist = plistlib.loads(output)

            for item in plist[0]['_items']:
                model = item.get('sppci_model', item.get('_name', 'Unknown GPU'))

                print(f"MPS GPU: {model}")
                print(f"Shared Unified Memory: {total_memory / (1024 ** 3):.2f} GB (system-wide)")
        except Exception as e:
            print(f"Failed to retrieve MPS info: {e}")
    else:
        print("MPS is not available.")

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

        if (pretrained in ['bert', 'roberta', 'distilbert', 'albert', 'xlnet', 'gpt2', 'llama', 'deepseek']):
            pt_type += 'attention:tokenized'

        elif (pretrained in ['glove', 'word2vec']):
            pt_type += 'co-occurrence:word'

        elif (pretrained in ['hyperbolic']):
            pt_type += 'hyperbolic:word'

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


def index_dataset(dataset, opt, pt_model=None, debug=False):
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
    print(f'\n\tindexing dataset.... dataset: {dataset}, vtype: {opt.vtype}, model: {opt.pretrained}, debug: {debug}')
    
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
    devel_index = index(dataset.devel_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary, opt, debug)
    test_index = index(dataset.test_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary, opt, debug)

    # Save the indexed dataset to a pickle file
    print(f"Saving indexed dataset to {pickle_file}...")
    with open(pickle_file, "wb") as f:
        pickle.dump((word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index), f)

    return word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index




def index(data, vocab, known_words, analyzer, unk_index, out_of_vocabulary, opt, debug=False):
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

    if (debug):
        # Write vocab words, known words, and unknown words to files
        with open(vocab_words_file, 'w') as vocab_file:
            vocab_file.write("\n".join(sorted(all_vocab_words)))

        with open(known_words_file, 'w') as known_file:
            known_file.write("\n".join(sorted(all_known_words)))

        with open(unknown_words_file, 'w') as unknown_file:
            unknown_file.write("\n".join(sorted(all_unknown_words)))

        print(f"Text and tokens written to: {text_and_tokens_file}")
        print(f"Vocabulary words written to: {vocab_words_file}")
        print(f"Known words written to: {known_words_file}")
        print(f"Unknown words written to: {unknown_words_file}")

    return indexed_docs



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




# ------------------------------------------------------------------------------------------------------------------------------------------------
#
# text preprocessing functions
#

def preprocess(
        text_series: pd.Series, 
        remove_punctuation: bool=True, 
        lowercase: bool=False, 
        remove_stopwords: bool=False, 
        remove_special_chars: bool=False, 
        array:  bool=False):
    """
    Preprocess a pandas Series of texts by removing punctuation, optionally lowercasing, 
    and optionally removing stopwords. Unmatched tokens, warnings, and unwanted patterns 
    are also removed.

    Parameters:
    - text_series: A pandas Series containing text data (strings).
    - remove_punctuation: Boolean indicating whether to remove punctuation.
    - lowercase: Boolean indicating whether to convert text to lowercase.
    - remove_stopwords: Boolean indicating whether to remove stopwords.
    - remove_special_chars: Boolean indicating whether to remove special LaTeX-like symbols. 
    - array: Boolean indicating whether to return as a NumPy array or as a list.

    Returns:
    - processed_texts: processed text strings, eitehr as list or numpy array (depending on array argument)
    """

    print("preprocessing...")
    print("text_series:", type(text_series), text_series.shape)
    
    # Load stop words once outside the loop
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    punctuation_table = str.maketrans('', '', string.punctuation)  # Translation table to remove punctuation

    # Regular expression for the specific pattern: '['^', '^', ...]...'
    # from arxiv data, post preprocessing
    unmatched_pattern = r"\[\s*'(\^)',?\s*(?:'(?:\^)',?\s*)*\]\.{3}"

    # Function to process each text
    def process_text(text):

        text = text.replace('\n', ' ')  # Replace \n characters with a space
        
        if lowercase:
            text = text.lower()

        # designed for arxiv data
        if remove_special_chars:
            text = re.sub(r'\$\{[^}]*\}|\$|\\[a-z]+|[{}]', ' ', text)               # Replace with space
            text = re.sub(unmatched_pattern, ' ', text)                             # Replace unmatched pattern with space
                
        if remove_punctuation:
            text = re.sub(rf"[{string.punctuation}]", ' ', text)                    # Replace punctuation with space

        if remove_stopwords:
            for stopword in stop_words:
                text = re.sub(r'\b' + re.escape(stopword) + r'\b', '', text)

        # Ensure extra spaces are removed after stopwords are deleted
        return ' '.join(text.split())

    # Track and warn about empty rows
    empty_row_indices = []

    def check_empty_and_process(text, index):
        processed = process_text(text)
        if not processed.strip():  # Check if the processed text is empty
            empty_row_indices.append((index, text))
        return processed

    # Use Parallel processing with multiple cores
    processed_texts = Parallel(n_jobs=-1)(
        delayed(check_empty_and_process)(text, idx) for idx, text in text_series.items()
    )

    # Print warnings for empty rows
    if empty_row_indices:
        print("[WARNING] The following rows are empty after preprocessing:")
        for idx, original_text in empty_row_indices:
            print(f"Row {idx}: '{original_text}'")

    if (array):
        # Return as a NumPy array
        return np.array(processed_texts)
    else:
        # Return as a list
        return list(processed_texts)



def _mask_numbers(data, number_mask='[NUM]'):
    """
    Masks numbers in the given text data with a placeholder.
    """
    mask = re.compile(r'\b[0-9][0-9.,-]*\b')
    return [mask.sub(number_mask, text) for text in data]


def preprocess_text(data):
    """
    Preprocess the text data by converting to lowercase, masking numbers, 
    removing punctuation, and removing stopwords.
    """
    import re
    from nltk.corpus import stopwords
    from string import punctuation

    stop_words = set(stopwords.words('english'))
    punct_table = str.maketrans("", "", punctuation)

    def _remove_punctuation_and_stopwords(data):
        """
        Removes punctuation and stopwords from the text data.
        """
        cleaned = []
        for text in data:
            # Remove punctuation and lowercase text
            text = text.translate(punct_table).lower()
            # Remove stopwords
            tokens = text.split()
            tokens = [word for word in tokens if word not in stop_words]
            cleaned.append(" ".join(tokens))
        return cleaned

    # Apply preprocessing steps
    masked = _mask_numbers(data)
    cleaned = _remove_punctuation_and_stopwords(masked)
    return cleaned
