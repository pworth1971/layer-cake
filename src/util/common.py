import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from tqdm import tqdm
import torch
from scipy.sparse import vstack, issparse
from joblib import Parallel, delayed
import multiprocessing
import itertools

from util.csv_log import CSVLog
import matplotlib.pyplot as plt

from embedding.pretrained import GloVe, BERT, Word2Vec, FastText


DEFAULT_BERT_PRETRAINED_MODEL = 'bert-base-uncased'

VECTOR_CACHE = '../.vector_cache'                       # assumes everything is run from /bin directory


# ---------------------------------------------------------------------------------------------------------------------------------------
# load_pretrained_embeddings()
#
# returns Boolean and then data structure with embeddings, None if 
# emb_model is not one if the acceptable values
#
# ---------------------------------------------------------------------------------------------------------------------------------------
def load_pretrained_embeddings(model, args):

    print("----- util.common.load_pretrained_embeddings() -----", {model})

    if model=='glove':
        print("path:", {args.glove_path})
        print("Loading GloVe...")
        return True, GloVe(path=args.glove_path)
    
    elif model=='word2vec':
        print("path:", {args.word2vec_path})
        print("Loading Word2Vec...")
        return True, Word2Vec(path=args.word2vec_path, limit=1000000)
    
    elif model=='fasttext':
        print("path:", {args.fasttext_path})
        print("Loading fasttext...")
        return True, FastText(path=args.fasttext_path, limit=1000000)
    
    elif model=='bert':
        print("path:", {args.bert_path})
        print("Loading BERT...")
        return True, BERT(model_name=DEFAULT_BERT_PRETRAINED_MODEL, emb_path=args.bert_path)

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


# ---------------------------------------------------------------------------------------------------------------------------

def index_dataset(dataset, pretrained=None):

    print()
    print(f"indexing dataset")
    
    # build the vocabulary
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
    print()

    return word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index

# ---------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------
# init_logile()
# 
# Simple, legacy log file that included run info in 'method'
#
def init_simple_logfile(method_name, opt):

    logfile = CSVLog(
        file=opt.log_file, 
        columns=['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'], 
        verbose=True, 
        overwrite=False)

    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', method_name)
    
    assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'
    
    return logfile



# ------------------------------------------------------------------------------------------------------------------------------------------
# init_layered_logile()
# 
# Enhanced log info to include explictly the model type, whether or not its 'supervised', ie added WC Embeddings, 
# as well as tuning parametr and whether or not pretraiend embeddings are used and if so what type
# 
def init_layered_logfile(method_name, pretrained, embeddings, opt):

    logfile = CSVLog(
        file=opt.log_file, 
        columns=[
            'dataset', 
            'model', 
            'pretrained', 
            'embeddings', 
            'wc-supervised', 
            'params', 
            'epoch', 
            'tunable',
            'measure', 
            'value', 
            'run', 
            'timelapse'], 
        verbose=True, 
        overwrite=False)

    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('model', opt.net)
    logfile.set_default('pretrained', pretrained)
    logfile.set_default('embeddings', embeddings)
    logfile.set_default('wc-supervised', opt.supervised)
    logfile.set_default('params', method_name)
    logfile.set_default('run', opt.seed)
    logfile.set_default('tunable', opt.tunable)

    #
    # TODO
    # adapt to layered log file defaults (more sophisticated handling)
    #
    #assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'

    return logfile


# ------------------------------------------------------------------------------------------------------------------------------------------
# init_layered_logile_svm()
# 
# Enhanced log info for svm program suport
#
def init_layered_logfile_svm(logfile, method_name, dataset, model, pretrained, embeddings, supervised):

    print()
    print("initializing SVM layered log file...")

    logfile = CSVLog(
        file=logfile, 
        columns=[
            'dataset', 
            'model', 
            'pretrained', 
            'embeddings', 
            'wc-supervised', 
            'params',
            'epoch',
            'tunable',
            'measure', 
            'run',
            'value',
            'timelapse'], 
        verbose=True, 
        overwrite=False)

    print("setting defaults...")
    print("embeddings:", embeddings)
    
    logfile.set_default('dataset', dataset)
    logfile.set_default('pretrained', pretrained)
    logfile.set_default('model', model)
    logfile.set_default('embeddings', embeddings)
    logfile.set_default('wc-supervised', supervised)
    logfile.set_default('params', method_name)

    # normalize data fields
    logfile.set_default('tunable', "NA")
    logfile.set_default('epoch', "NA")
    logfile.set_default('run', "NA")

    #
    # TODO
    # adapt to layered log file defaults (more sophisticated handling)
    #
    #assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'

    return logfile