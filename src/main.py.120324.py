import argparse
import os
import numpy as np
from time import time
import scipy
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from embedding.supervised import get_supervised_embeddings, STWFUNCTIONS
from embedding.pretrained import GloVeEmbeddings, Word2VecEmbeddings, FastTextEmbeddings, BERTEmbeddings
#from embedding.pretrained import RoBERTaEmbeddings, DistilBERTEmbeddings, XLNetEmbeddings, GPT2Embeddings, LlamaEmbeddings

from model.classification import NeuralClassifier

from data.dataset import *

from util.common import *
from util.early_stop import EarlyStopping
from util.csv_log import CSVLog
from util.file import create_if_not_exist
from util.metrics import *


VECTOR_CACHE = "../.vector_cache"                               # cache directory for pretrained models



def init_Net(nC, vocabsize, pretrained_embeddings, sup_range, device, opt):

    print("initializing neural model...")

    net_type=opt.net
    hidden = opt.channels if net_type == 'cnn' else opt.hidden
    if opt.droptype == 'sup':
        drop_range = sup_range
    elif opt.droptype == 'learn':
        drop_range = [pretrained_embeddings.shape[1], pretrained_embeddings.shape[1]+opt.learnable]
    elif opt.droptype == 'none':
        drop_range = None
    elif opt.droptype == 'full':
        drop_range = [0, pretrained_embeddings.shape[1]+opt.learnable]

    print('droptype =', opt.droptype)
    print('droprange =', drop_range)
    print('dropprob =', opt.dropprob)

    model = NeuralClassifier(
        net_type,
        output_size=nC,
        hidden_size=hidden,
        vocab_size=vocabsize,
        learnable_length=opt.learnable,
        pretrained=pretrained_embeddings,
        drop_embedding_range=drop_range,
        drop_embedding_prop=opt.dropprob)

    model.xavier_uniform()
    model = model.to(device)
    if opt.tunable:
        model.finetune_pretrained()

    embsizeX, embsizeY = model.get_embedding_size()
    #print("embsizeX:", embsizeX)
    #print("embsizeY:", embsizeY)

    lrnsizeX, lrnsizeY = model.get_learnable_embedding_size()
    #print("lrnsizeX:", lrnsizeX)
    #print("lrnsizeY:", lrnsizeY)

    return model, embsizeX, embsizeY, lrnsizeX, lrnsizeY


def set_method_name(opt):
    method_name = opt.net

    if opt.pretrained:
        method_name += f'-{opt.pretrained}'
    if opt.learnable > 0:
        method_name += f'-learn{opt.learnable}'
    if opt.supervised:
        sup_drop = 0 if opt.droptype != 'sup' else opt.dropprob
        method_name += f'-supervised-d{sup_drop}-{opt.supervised_method}'
    if opt.dropprob > 0:
        if opt.droptype != 'sup':
            method_name += f'-Drop{opt.droptype}{opt.dropprob}'
    if (opt.pretrained or opt.supervised) and opt.tunable:
        method_name+='-tunable'
    if opt.weight_decay > 0:
        method_name+=f'_wd{opt.weight_decay}'
    if opt.net in {'lstm', 'attn'}:
        method_name+=f'-h{opt.hidden}'
    if opt.net== 'cnn':
        method_name+=f'-ch{opt.channels}'
    return method_name


def index_dataset_new(opt, dataset, pt_model=None, debug=False):
    """
    Indexes the dataset for use with word-based (e.g., GloVe, Word2Vec, FastText)
    and token-based (e.g., BERT, RoBERTa, DistilBERT) embeddings.

    Parameters:
    ----------
    dataset : Dataset
        The dataset object containing raw text and a fitted vectorizer.
    pt_model : PretrainedEmbeddings class (instantiated) optional
        Pretrained embedding object to extend the known vocabulary.
    debug : bool, optional

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

    # Build the vocabulary
    word2index = dict(dataset.vocabulary)
    known_words = set(word2index.keys())
    if pt_model is not None:
        print(f'updating known_words with pretrained.vocabulary(): {type(pt_model.vocabulary())}; {len(pt_model.vocabulary())}')
        known_words.update(pt_model.vocabulary())

    # Add special tokens
    word2index['UNKTOKEN'] = len(word2index)
    word2index['PADTOKEN'] = len(word2index)
    unk_index = word2index['UNKTOKEN']
    pad_index = word2index['PADTOKEN']

    # Prepare analyzer (handles both default and custom tokenizers)
    analyzer = dataset.analyzer()
    print("analyzer:\n", analyzer)
              
    # Index documents
    out_of_vocabulary = dict()
    devel_index = index_new(opt, dataset.devel_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary, pt_model.get_model(), suffix='dev', debug=debug)
    test_index = index_new(opt, dataset.test_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary, pt_model.get_model(), suffix='test', debug=debug)

    return word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index



def index_new(opt, data, vocab, known_words, analyzer, unk_index, out_of_vocabulary, model_name, suffix, debug=False):
    """
    Indexes a list of string documents using the provided analyzer and logs tokens into files.

    Parameters:
    ----------
    opt : Namespace
        Contains configurations like `pretrained` and `dataset`.
    data : list of str
        List of documents to be indexed.
    vocab : dict
        Fixed mapping [str] -> [int] for tokens.
    known_words : set
        Set of known tokens (e.g., words or subwords in pretrained embeddings).
    analyzer : callable
        Function to tokenize the documents (default tokenizer or custom tokenizer for BERT-like models).
    unk_index : int
        Index of the unknown token.
    out_of_vocabulary : dict
        Mapping of OOV tokens to indices.
    model_name : str
        Name of the pretrained model.
    debug : bool, optional
        Flag to enable detailed debugging.

    Returns:
    -------
    list of lists
        Indexed version of the documents.
    """
    print(f'\n\tindexing documents...')

    # Paths for logging tokens
    os.makedirs(LOG_DIR, exist_ok=True)

    file_name = f"{opt.dataset}.{opt.pretrained}.{model_name}.{suffix}"
    in_vocab_file = os.path.join(LOG_DIR, f"{file_name}.in_vocab")
    oov_file = os.path.join(LOG_DIR, f"{file_name}.oov")
    unknown_file = os.path.join(LOG_DIR, f"{file_name}.unknown")
    parsed_text_file = os.path.join(LOG_DIR, f"{file_name}.parsed_text")

    with open(in_vocab_file, 'w') as in_vocab_log, \
         open(oov_file, 'w') as oov_log, \
         open(unknown_file, 'w') as unknown_log, \
         open(parsed_text_file, 'w') as parsed_text_log:

        indexes = []
        vocabsize = len(vocab)
        unk_count = 0
        knw_count = 0
        out_count = 0

        # Set to keep track of unique tokens for in_vocab
        unique_in_vocab_tokens = set()

        pbar = tqdm(data, desc=f'indexing documents...')
        for text in pbar:
            if debug:
                print("text:", text)
            tokens = analyzer(text)  # Tokenize the text
            index = []

            # Log the text and tokens
            parsed_text_log.write(f"Text: {text}\nTokens: {', '.join(tokens)}\n\n")

            for token in tokens:
                if token in vocab:
                    if debug:
                        print(f'token {token} in vocab...')
                    idx = vocab[token]

                    # Log only unique tokens
                    if token not in unique_in_vocab_tokens:
                        in_vocab_log.write(f"{token}\n")
                        unique_in_vocab_tokens.add(token)
                        
                elif token in known_words:
                    if debug:
                        print(f'token {token} in known_words...')
                    if token not in out_of_vocabulary:
                        if debug:
                            print(f'token {token} not in out_of_vocabulary...')
                        out_of_vocabulary[token] = vocabsize + len(out_of_vocabulary)
                    idx = out_of_vocabulary[token]
                    out_count += 1
                    oov_log.write(f"{token}\n")  # Log OOV token
                else:
                    if debug:
                        print(f'token {token} not in vocab or known_words...')
                    idx = unk_index
                    unk_count += 1
                    unknown_log.write(f"{token}\n")  # Log unknown token
                index.append(idx)
            indexes.append(index)
            knw_count += len(index)

            pbar.set_description(f'[UNK = {unk_count}/{knw_count} = {(100. * unk_count / knw_count):.2f}%]'
                                 f'[OOV = {out_count}/{knw_count} = {(100. * out_count / knw_count):.2f}%]')

    """
    print(f"\nToken logging completed:")
    print(f"- In vocabulary tokens: {in_vocab_file}")
    print(f"- Out-of-vocabulary tokens: {oov_file}")
    print(f"- Unknown tokens: {unknown_file}")
    print(f"- Parsed text and tokens: {parsed_text_file}")
    """

    return indexes



def index_new_old(data, vocab, known_words, analyzer, unk_index, out_of_vocabulary, debug=False):
    """
    Indexes a list of string documents using the provided analyzer.

    Parameters:
    ----------
    data : list of str
        List of documents to be indexed.
    vocab : dict
        Fixed mapping [str] -> [int] for tokens.
    known_words : set
        Set of known tokens (e.g., words or subwords in pretrained embeddings).
    analyzer : callable
        Function to tokenize the documents (default tokenizer or custom tokenizer for BERT-like models).
    unk_index : int
        Index of the unknown token.
    out_of_vocabulary : dict
        Mapping of OOV tokens to indices.
    debug : bool, optional

    Returns:
    -------
    list of lists
        Indexed version of the documents.
    """

    print(f'\n\tindexing documents...')

    indexes = []
    vocabsize = len(vocab)
    unk_count = 0
    knw_count = 0
    out_count = 0

    pbar = tqdm(data, desc=f'indexing documents...')
    for text in pbar:
        if debug: 
            print("text:", text)
        tokens = analyzer(text)             # Tokenize the text
        index = []
        for token in tokens:
            if token in vocab:
                if debug:
                    print(f'token {token} in vocab...')
                idx = vocab[token]
            elif token in known_words:
                if debug:
                    print(f'token {token} in known_words...')
                if token not in out_of_vocabulary:
                    print(f'token {token} not in out_of_vocabulary...')
                    out_of_vocabulary[token] = vocabsize + len(out_of_vocabulary)
                idx = out_of_vocabulary[token]
                out_count += 1
            else:
                if debug:
                    print(f'token {token} not in vocab or known_words...')
                idx = unk_index
                unk_count += 1
            index.append(idx)
        indexes.append(index)
        knw_count += len(index)

        pbar.set_description(f'[UNK = {unk_count}/{knw_count} = {(100. * unk_count / knw_count):.2f}%]'
                             f'[OOV = {out_count}/{knw_count} = {(100. * out_count / knw_count):.2f}%]')
    
    return indexes




def embedding_matrix(dataset, pretrained, vocabsize, word2index, out_of_vocabulary, opt):
    """
    Constructs an embedding matrix using pretrained embeddings (e.g., BERT) 
    and optionally supervised embeddings derived from the dataset.

    Parameters:
    ----------
    dataset : Dataset
        The dataset object containing raw text, label matrices, and vectorized data.
    pretrained : PretrainedEmbeddings
        The pretrained embedding object (e.g., BERTEmbeddings).
    vocabsize : int
        Size of the vocabulary, including special tokens (e.g., UNK and PAD).
    word2index : dict
        Mapping of words or tokens to their respective indices.
    out_of_vocabulary : dict
        Mapping of out-of-vocabulary tokens to indices beyond the standard vocabulary.
    opt : argparse.Namespace
        Configuration object containing options for:
        - pretrained embeddings (`opt.pretrained`).
        - supervised embeddings (`opt.supervised`).
        - supervised method (`opt.supervised_method`).
        - maximum label space (`opt.max_label_space`).
        - whether to z-score normalize (`opt.nozscore`).

    Returns:
    -------
    tuple:
        - pretrained_embeddings (torch.Tensor or None): 
            The final embedding matrix of shape (vocabsize, embedding_dim), 
            combining pretrained and supervised embeddings if applicable.
        - sup_range (list or None): 
            A list indicating the range of supervised embeddings in the final matrix, 
            or `None` if no supervised embeddings are included.

    Notes:
    -----
    - Pretrained embeddings are extracted using the `extract` method of the `pretrained` object.
    - Supervised embeddings are calculated using `get_supervised_embeddings`.
    - Tokens not found in the pretrained vocabulary are handled as OOV.

    Example:
    -------
    pretrained_embeddings, sup_range = embedding_matrix(
        dataset=dataset,
        pretrained=bert_embeddings,
        vocabsize=12835,
        word2index=word2index,
        out_of_vocabulary=oov_dict,
        opt=options
    )
    """
    
    print('\n\tcomputing embedding matrix...')
    
    pretrained_embeddings = None
    sup_range = None
    
    if opt.pretrained or opt.supervised:

        pretrained_embeddings = []

        if pretrained is not None:
            word_list = get_word_list(word2index, out_of_vocabulary)
            weights = pretrained.extract(word_list)
            pretrained_embeddings.append(weights)
            print('\t[pretrained-matrix] ', weights.shape)
            del pretrained

        if opt.supervised:
            Xtr, _ = dataset.vectorize()
            Ytr = dataset.devel_labelmatrix
            F = get_supervised_embeddings(Xtr, Ytr,
                                          method=opt.supervised_method,
                                          max_label_space=opt.max_label_space,
                                          dozscore=(not opt.nozscore))
            num_missing_rows = vocabsize - F.shape[0]
            F = np.vstack((F, np.zeros(shape=(num_missing_rows, F.shape[1]))))
            F = torch.from_numpy(F).float()
            print('\t[supervised-matrix]', F.shape)

            offset = 0
            if pretrained_embeddings:
                offset = pretrained_embeddings[0].shape[1]
            sup_range = [offset, offset + F.shape[1]]
            pretrained_embeddings.append(F)

        # Step 3: Concatenate Pretrained and Supervised Embeddings
        pretrained_embeddings = [tensor.to(opt.device) for tensor in pretrained_embeddings]  # Ensure all tensors are on the same device
        pretrained_embeddings = torch.cat(pretrained_embeddings, dim=1)
        
    #print('[embedding matrix done]')
    return pretrained_embeddings, sup_range


def init_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)


def init_logfile(method_name, opt):
    logfile = CSVLog(opt.log_file, ['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'])
    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', method_name)
    assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'
    return logfile



def init_loss(classification_type, device, ytr=None):
    """
    Initialize the loss function based on the classification type.
    
    Parameters:
    ----------
    classification_type : str
        The type of classification problem ('multilabel' or 'singlelabel').
    device : torch.device
        The device to which the loss function will be moved.
    ytr : numpy.ndarray or None, optional
        Training labels for computing class weights in case of class imbalance.
        Should be provided for both single-label and multi-label cases.

    Returns:
    -------
    torch.nn.Module
        Initialized loss function.
    """
    assert classification_type in ['multilabel', 'multi-label', 'single-label', 'singlelabel'], 'unknown classification type'

    print(f'initializing loss function for {classification_type} classification...')
    
    if ytr is not None:
        
        print("ytr:", type(ytr), ytr.shape)
        
        print("computing class weights...")
        
        if classification_type in ['multilabel', 'multi-label']:
            # Compute class weights for multi-label data
            class_counts = np.sum(ytr, axis=0)  # Sum along columns
            class_weights = class_counts.max() / (class_counts + 1e-6)  # Avoid division by zero
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
        else:  # single-label
            # Compute class weights for single-label data
            ytr_labels = np.argmax(ytr, axis=1)  # Convert one-hot to labels if needed
            class_weights = compute_class_weight('balanced', classes=np.unique(ytr_labels), y=ytr_labels)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
            loss = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        # Default loss without class weights
        if classification_type in ['multilabel', 'multi-label']:
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = torch.nn.CrossEntropyLoss()

    return loss.to(device)



def init_loss_old(classification_type, device):
    """
    assert classification_type in ['multilabel','singlelabel'], 'unknown classification mode'
    
    L = torch.nn.BCEWithLogitsLoss() if classification_type == 'multilabel' else torch.nn.CrossEntropyLoss()
    return L.cuda()
    """

    assert classification_type in ['multilabel', 'multi-label', 'single-label', 'singlelabel'], 'unknown classification type'

    # In your criterion (loss function) initialization, pass the class weights
    if classification_type == 'multilabel':
        L = torch.nn.BCEWithLogitsLoss()
    else:
        L = torch.nn.CrossEntropyLoss()
        
    return L.to(device)



def train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name, opt):

    print("training model...")

    as_long = isinstance(criterion, torch.nn.CrossEntropyLoss)
    loss_history = []
    if opt.max_epoch_length is not None: # consider an epoch over after max_epoch_length
        tr_len = len(train_index)
        train_for = opt.max_epoch_length*opt.batch_size
        from_, to_= (epoch*train_for) % tr_len, ((epoch+1)*train_for) % tr_len
        print(f'epoch from {from_} to {to_}')
        if to_ < from_:
            train_index = train_index[from_:] + train_index[:to_]
            if issparse(ytr):
                ytr = np.vstack((ytr[from_:], ytr[:to_]))
            else:
                ytr = np.concatenate((ytr[from_:], ytr[:to_]))
        else:
            train_index = train_index[from_:to_]
            ytr = ytr[from_:to_]

    model.train()
    for idx, (batch, target) in enumerate(batchify_legacy(train_index, ytr, opt.batch_size, pad_index, opt.device, as_long)):
        optim.zero_grad()
        loss = criterion(model(batch), target)
        loss.backward()
        clip_gradient(model)
        optim.step()
        loss_history.append(loss.item())

        if idx % opt.log_interval == 0:
            interval_loss = np.mean(loss_history[-opt.log_interval:])
            print(f'{opt.dataset} {method_name} Epoch: {epoch}, Step: {idx}, Training Loss: {interval_loss:.6f}')

    mean_loss = np.mean(interval_loss)
    #logfile.add_row(epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time() - tinit)
    return mean_loss



def test(model, test_index, yte, pad_index, classification_type, tinit, epoch, logfile, criterion, measure_prefix, opt, embedding_size):

    print("testing model...")

    model.eval()
    predictions = []
    target_long = isinstance(criterion, torch.nn.CrossEntropyLoss)
    for batch, target in tqdm(
            batchify_legacy(test_index, yte, opt.batch_size_test, pad_index, opt.device, target_long=target_long),
            desc='evaluation: '
    ):
        print("batch:", type(batch), batch.shape)
        print("target:", type(target), target.shape)

        logits = model(batch)

        # Ensure logits and target sizes match
        if logits.size(0) != target.size(0):
            min_size = min(logits.size(0), target.size(0))
            logits = logits[:min_size]
            target = target[:min_size]

        loss = criterion(logits, target).item()
        prediction = csr_matrix(predict(logits, classification_type=classification_type))
        predictions.append(prediction)

    yte_ = scipy.sparse.vstack(predictions)
    
    Mf1, mf1, acc = evaluation_legacy(yte, yte_, classification_type)
    print(f'[{measure_prefix}] Macro-F1={Mf1:.4f} Micro-F1={mf1:.4f} Accuracy={acc:.4f}')
    
    tend = time() - tinit

    if (measure_prefix == 'final-te'):

        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

        """
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)
        """

    return Mf1, mf1, acc, yte, yte_




def load_pt_model(opt):
    
    print("loading pretrained embeddings...")
    
    if opt.pretrained == 'glove':
        return GloVeEmbeddings(path=opt.glove_path)
    elif opt.pretrained == 'word2vec':
        return Word2VecEmbeddings(path=opt.word2vec_path, limit=1000000)
    elif opt.pretrained == 'fasttext':
        return FastTextEmbeddings(path=opt.fasttext_path, limit=1000000)
    elif opt.pretrained == 'bert':
        return BERTEmbeddings(path=opt.bert_path)
    elif opt.pretrained == 'roberta':
        return RoBERTaEmbeddings(path=opt.roberta_path)
    elif opt.pretrained == 'distilbert':
        return DistilBERTEmbeddings(path=opt.distilbert_path)
    elif opt.pretrained == 'xlnet':
        return XLNetEmbeddings(path=opt.xlnet_path)
    elif opt.pretrained == 'gpt2':
        return GPT2Embeddings(path=opt.gpt2_path)
    elif opt.pretrained == 'llama':
        return LlamaEmbeddings(path=opt.llama_path)
    
    return None




def main(opt):

    #method_name = set_method_name(opt)
    #print("method_name", method_name)
    #logfile = init_logfile(method_name, opt)

    # initialize logging and other system run variables
    already_modelled, logfile, method_name, pretrained, embeddings, embedding_type, emb_path, lm_type, mode, system = initialize_testing(opt)

    # check to see if model params have been computed already
    if (already_modelled and not opt.force):
        print(f'--- model {method_name} with embeddings {embeddings}, pretrained == {pretrained}, tunable == {opt.tunable}, and wc_supervised == {opt.supervised} for {opt.dataset} already calculated, run with --force option to override. ---')
        exit(0)

    print("method_name:", method_name)
    print("pretrained:", pretrained)
    print("embeddings:", embeddings)    
    print("embedding_type:", embedding_type)
    print("emb_path:", emb_path)
    print("lm_type:", lm_type)
    print("mode:", mode)
    print("system:", system)

    pt_model = load_pt_model(opt)
    print("pt_model:", pt_model)

    dataset = Dataset.load(
        dataset_name=opt.dataset, 
        vtype=opt.vtype,
        pt_model=pt_model,
        pickle_dir=opt.pickle_dir 
    ).show()

    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset_new(opt, dataset=dataset, pt_model=pt_model)
    print("word2index", type(word2index), len(word2index))
    print("out_of_vocabulary", type(out_of_vocabulary), len(out_of_vocabulary))
    print("devel_index", type(devel_index), len(devel_index))
    print("test_index", type(test_index), len(test_index))

    # dataset split tr/val/test
    val_size = min(int(len(devel_index) * .2), 20000)
    train_index, val_index, ytr, yval = train_test_split(
        devel_index, dataset.devel_target, test_size=val_size, random_state=opt.seed, shuffle=True
    )
    print("train_index", type(train_index), len(train_index))
    print("val_index", type(val_index), len(val_index))

    yte = dataset.test_target
    print("ytr", type(ytr), ytr.shape)
    print("yval", type(yval), yval.shape)
    print("yte", type(yte), yte.shape)

    target_names = dataset.target_names
    print("target_names", type(target_names), len(target_names))

    vocabsize = len(word2index) + len(out_of_vocabulary)
    print("vocabsize", vocabsize)

    pretrained_embeddings, sup_range = embedding_matrix(dataset, pt_model, vocabsize, word2index, out_of_vocabulary, opt)
    print("pretrained_embeddings", type(pretrained_embeddings), pretrained_embeddings.shape)
    print("sup_range", type(sup_range), sup_range)
          
    lc_model, embedding_sizeX, embedding_sizeY, lrn_sizeX, lrn_sizeY = init_Net(dataset.nC, vocabsize, pretrained_embeddings, sup_range, opt.device, opt)
    print("lc_model:\n", lc_model)

    optim = init_optimizer(lc_model, lr=opt.lr, weight_decay=opt.weight_decay)
    print("optim:\n", optim)

    # Initialize loss with class weights
    criterion = init_loss(
        classification_type=dataset.classification_type,
        device=opt.device,
        ytr=None
        #ytr=ytr                                     # Pass the training labels for class weight computation
    )
    print("criterion:\n", criterion)
    
    #
    # establish dimensions (really shapes) of embedding and learning layers for logging
    #    
    emb_size_str = f'({embedding_sizeX}, {embedding_sizeY})'
    #print("emb_size:", emb_size_str)
    lrn_size_str = f'({lrn_sizeX}, {lrn_sizeY})'
    #print("lrn_size:", lrn_size_str)
    
    emb_size_str = f'{emb_size_str}:{lrn_size_str}'
    print("emb_size_str:", emb_size_str)

    # train-validate
    tinit = time()
    
    create_if_not_exist(opt.checkpoint_dir)
    
    early_stop = EarlyStopping(
        lc_model, 
        patience=opt.patience, 
        checkpoint=f'{opt.checkpoint_dir}/{opt.net}-{opt.dataset}'
    )

    for epoch in range(1, opt.nepochs + 1):
        train(lc_model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name, opt)

        # validation
        macrof1, microf1, acc, yte, yte_ = test(lc_model, val_index, yval, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'va', opt, embedding_size=emb_size_str)
        early_stop(macrof1, microf1, epoch)
        if opt.test_each>0:
            if (opt.plotmode and (epoch==1 or epoch%opt.test_each==0)) or (not opt.plotmode and epoch%opt.test_each==0 and epoch<opt.nepochs):
                test(lc_model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'te', opt, embedding_size=emb_size_str)

        if early_stop.STOP:
            print('[early-stop]')
            if not opt.plotmode: # with plotmode activated, early-stop is ignored
                break

    # restores the best model according to the Mf1 of the validation set (only when plotmode==False)
    stoptime = early_stop.stop_time - tinit
    stopepoch = early_stop.best_epoch
    #logfile.add_row(epoch=stopepoch, measure=f'early-stop', value=early_stop.best_score, timelapse=stoptime)

    if not opt.plotmode:
        print('performing final evaluation')
        res_model = early_stop.restore_checkpoint()
        print("res_model:\n", res_model)

        """
        if opt.val_epochs>0:
            print(f'last {opt.val_epochs} epochs on the validation set')
            for val_epoch in range(1, opt.val_epochs + 1):
                train(res_model, val_index, yval, pad_index, tinit, logfile, criterion, optim, epoch+val_epoch, method_name, opt)
        """

        # test
        print('Training complete: testing')
        Mf1, mf1, acc, yte, yte_ = test(res_model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'final-te', opt, embedding_size=emb_size_str)

        print(classification_report(yte, yte_, target_names=target_names, digits=4))

        print("\n\t--Final Layer Cake Metrics--")
        #print(f"Macro-F1 = {Mf1:.4f}, Micro-F1 = {mf1:.4f}, Accuracy = {acc:.4f}, H-loss = {h_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, Jaccard = {j_index:.4f}")
        print(f"Macro-F1 = {Mf1:.4f}, Micro-F1 = {mf1:.4f}, Accuracy = {acc:.4f}")
    



if __name__ == '__main__':

    print("\n\tLAYER CAKE: Neural text classification with Word-Class Embeddings...")

    available_datasets = Dataset.dataset_available
    print("available_datasets", available_datasets)

    available_dropouts = {'sup','none','full','learn'}

    # Training settings
    parser = argparse.ArgumentParser(description='Neural text classification with Word-Class Embeddings')
    parser.add_argument('--dataset', type=str, default='reuters21578', metavar='str',
                        help=f'dataset, one in {available_datasets}')
    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', 
                        help=f'dataset base vectorization strategy, in [tfidf, count]')
    
    parser.add_argument('--batch-size', type=int, default=50, metavar='int',
                        help='input batch size (default: 50)')
    parser.add_argument('--batch-size-test', type=int, default=125, metavar='int',
                        help='batch size for testing (default: 125)')
    parser.add_argument('--nepochs', type=int, default=150, metavar='int',
                        help='number of epochs (default: 150)')
    parser.add_argument('--patience', type=int, default=5, metavar='int',
                        help='patience for early-stop (default: 5)')
    parser.add_argument('--plotmode', action='store_true', default=False,
                        help='in plot mode, executes a long run in order to generate enough data to produce trend plots'
                             ' (test-each should be >0. This mode is used to produce plots, and does not perform a '
                             'final evaluation on the test set other than those performed after test-each epochs).')
    parser.add_argument('--hidden', type=int, default=512, metavar='int',
                        help='hidden lstm size (default: 512)')
    parser.add_argument('--channels', type=int, default=256, metavar='int',
                        help='number of cnn out-channels (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='float',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='float',
                        help='weight decay (default: 0)')
    parser.add_argument('--droptype', type=str, default='sup', metavar='DROPTYPE',
                        help=f'chooses the type of dropout to apply after the embedding layer. Default is "sup" which '
                             f'only applies to word-class embeddings (if present). Other options include "none" which '
                             f'does not apply dropout (same as "sup" with no supervised embeddings), "full" which '
                             f'applies dropout to the entire embedding, or "learn" that applies dropout only to the '
                             f'learnable embedding.')
    parser.add_argument('--dropprob', type=float, default=0.2, metavar='[0.0, 1.0]',
                        help='dropout probability (default: 0.2)')
    parser.add_argument('--seed', type=int, default=1, metavar='int',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='int',
                        help='how many batches to wait before printing training status')
    parser.add_argument('--log-file', type=str, default='../log/legacy.tsv', metavar='str',
                        help='path to the log csv file')
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str',
                        help=f'if set, specifies the path where to save/load the dataset pickled (set to None if you '
                             f'prefer not to retain the pickle file)')
    parser.add_argument('--test-each', type=int, default=0, metavar='int',
                        help='how many epochs to wait before invoking test (default: 0, only at the end)')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoint', metavar='str',
                        help='path to the directory containing checkpoints')
    parser.add_argument('--net', type=str, default='lstm', metavar='str',
                        help=f'net, one in {NeuralClassifier.ALLOWED_NETS}')
    parser.add_argument('--learnable', type=int, default=0, metavar='int',
                        help='dimension of the learnable embeddings (default 0)')
    parser.add_argument('--val-epochs', type=int, default=1, metavar='int',
                        help='number of training epochs to perform on the validation set once training is '
                             'over (default 1)')
    
    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext',
                        help='pretrained embeddings, use "glove", "word2vec", or "fasttext" (default None)')
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use supervised embeddings')
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    parser.add_argument('--glove-path', type=str, default=VECTOR_CACHE+'/GloVe',
                        metavar='PATH',
                        help=f'path to glove.840B.300d pretrained vectors (used only with --pretrained glove)')
    parser.add_argument('--word2vec-path', type=str, default=VECTOR_CACHE+'/Word2Vec/GoogleNews-vectors-negative300.bin',
                        metavar='str',
                        help=f'path to GoogleNews-vectors-negative300.bin pretrained vectors (used only '
                             f'with --pretrained word2vec)')
    parser.add_argument('--fasttext-path', type=str, default=VECTOR_CACHE+'/fastText/crawl-300d-2M.vec',
                        help=f'path to crawl-300d-2M.vec pretrained vectors (used only with --pretrained fasttext)')
    parser.add_argument('--bert-path', type=str, default=VECTOR_CACHE+'/BERT',
                        metavar='PATH',
                        help=f'Directory to BERT pretrained vectors, defaults to {VECTOR_CACHE}/BERT. Used only with --pretrained bert')
    parser.add_argument('--roberta-path', type=str, default=VECTOR_CACHE+'/RoBERTa',
                        metavar='PATH',
                        help=f'Directory to RoBERTa pretrained vectors, defaults to {VECTOR_CACHE}/RoBERTA. Used only with --pretrained roberta')
    parser.add_argument('--distilbert-path', type=str, default=VECTOR_CACHE+'/DistilBERT',
                        metavar='PATH',
                        help=f'Directory to DistilBERT pretrained vectors, defaults to {VECTOR_CACHE}/DistilBERT. Used only with --pretrained distilbert')
    parser.add_argument('--xlnet-path', type=str, default=VECTOR_CACHE+'/XLNet',
                        metavar='PATH',
                        help=f'Directory to XLNet pretrained vectors, defaults to {VECTOR_CACHE}/XLNet. Used only with --pretrained xlnet.')
    parser.add_argument('--gpt2-path', type=str, default=VECTOR_CACHE+'/GPT2',
                        metavar='PATH',
                        help=f'Directory to GPT2 pretrained vectors, defaults to {VECTOR_CACHE}/GPT2. Used only with --pretrained gpt2')
    parser.add_argument('--llama-path', type=str, default=VECTOR_CACHE+'/LLaMA',
                        metavar='PATH',
                        help=f'Directory to LLaMA pretrained vectors, defaults to {VECTOR_CACHE}/LlaMa. Used only with --pretrained llama')
    
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding (if larger, then PCA with this '
                             'number of components is applied (default 300)')
    parser.add_argument('--max-epoch-length', type=int, default=None, metavar='int',
                        help='number of (batched) training steps before considering an epoch over (None: full epoch)') #300 for wipo-sl-sc
    parser.add_argument('--force', action='store_true', default=False,
                        help='do not check if this experiment has already been run')
    parser.add_argument('--tunable', action='store_true', default=False,
                        help='pretrained embeddings are tunable from the beginning (default False, i.e., static)')
    parser.add_argument('--nozscore', action='store_true', default=False,
                        help='disables z-scoring form the computation of WCE')

    opt = parser.parse_args()

    """
    opt.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'running on {opt.device}')
    assert f'{opt.device}'=='cuda', 'forced cuda device but cpu found'
    """
    
    # Setup device prioritizing CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        opt.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        opt.device = torch.device("mps")
    else:
        opt.device = torch.device("cpu")
    print(f'running on {opt.device}')

    torch.manual_seed(opt.seed)

    assert opt.dataset in available_datasets, \
        f'unknown dataset {opt.dataset}'
    assert opt.pretrained in [None]+SUPPORTED_LMS, \
        f'unknown pretrained set {opt.pretrained}'
    assert not opt.plotmode or opt.test_each > 0, \
        'plot mode implies --test-each>0'
    assert opt.supervised_method in STWFUNCTIONS, \
        f'unknown supervised term weighting function; allowed are {STWFUNCTIONS}'
    assert opt.droptype in available_dropouts, \
        f'unknown dropout type; allowed are {available_dropouts}'
    if opt.droptype == 'sup' and opt.supervised==False:
        opt.droptype = 'none'
        print('warning: droptype="sup" but supervised="False"; the droptype changed to "none"')
    if opt.droptype == 'learn' and opt.learnable==0:
        opt.droptype = 'none'
        print('warning: droptype="learn" but learnable=0; the droptype changed to "none"')

    print("opt:", opt)

    main(opt)