import argparse

from time import time

import scipy
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from data.dataset import *

from embedding.pretrained import *
from embedding.supervised import get_supervised_embeddings, STWFUNCTIONS

from model.classification import NeuralClassifier, BertWCEClassifier
from model.classification import Token2BertEmbeddings, Token2WCEmbeddings

from util.early_stop import EarlyStopping
from util.common import *
from util.csv_log import CSVLog
from util.file import create_if_not_exist
from util.metrics import *




VECTOR_CACHE = '../.vector_cache'

# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 8
DEFAULT_GPU_BATCH_SIZE = 8
DEFAULT_MPS_BATCH_SIZE = 8



def init_Net(nC, vocabsize, pretrained_embeddings, sup_range, opt):

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
    model = model.to(opt.device)
    if opt.tunable:
        model.finetune_pretrained()

    embsizeX, embsizeY = model.get_embedding_size()
    #print("embsizeX:", embsizeX)
    #print("embsizeY:", embsizeY)

    lrnsizeX, lrnsizeY = model.get_learnable_embedding_size()
    #print("lrnsizeX:", lrnsizeX)
    #print("lrnsizeY:", lrnsizeY)

    return model, embsizeX, embsizeY, lrnsizeX, lrnsizeY


def init_Net_Bert(nC, vocabsize, WCE, sup_range, opt):
    """
    Initializes the BertWCEClassifier model with BERT embeddings and optional WCE embeddings.

    Parameters:
    ----------
    nC : int
        Number of output classes.
    vocabsize : int
        Size of the vocabulary.
    WCE : Word-Class Embeddings matrix or None.
    sup_range : list or None
        Range of supervised embeddings.
    opt : argparse.Namespace
        Command-line options/configuration.

    Returns:
    -------
    tuple : (model, embsizeX, embsizeY, lrnsizeX, lrnsizeY)
        - model: The initialized BertWCEClassifier instance.
        - embsizeX: Embedding size X (BERT embedding dimension).
        - embsizeY: Embedding size Y (WCE embedding dimension or 0).
        - lrnsizeX, lrnsizeY: Learnable embedding sizes (if applicable, otherwise 0).
    """
    print("Initializing BERT-based neural model...")

    # Determine model type
    net_type = opt.net
    hidden = opt.channels if net_type == 'cnn' else opt.hidden

    # Determine model name
    if opt.pretrained == 'bert':
        model_name = BERT_MODEL
    elif opt.pretrained == 'roberta':
        model_name = ROBERTA_MODEL
    elif opt.pretrained == 'distilbert':
        model_name = DISTILBERT_MODEL
    else:
        raise ValueError(f"Unsupported pretrained model type: {opt.pretrained}")

    # Initialize token-based BERT embeddings
    token2bert_embeddings = Token2BertEmbeddings(
        pretrained_model_name=model_name,
        device=opt.device
    )

    # Allow fine-tuning of BERT embeddings if tunable
    if opt.tunable:
        print("Setting BERT embeddings to tunable mode...")
        token2bert_embeddings.model.train()

    # Initialize optional Word-Class Embeddings (WCE)
    token2wce_embeddings = None
    if opt.supervised and WCE is not None:
        WCE_vocab = {key: idx for idx, key in enumerate(range(vocabsize))}
        token2wce_embeddings = Token2WCEmbeddings(
            WCE=WCE,
            WCE_range=sup_range,
            WCE_vocab=WCE_vocab,
            drop_embedding_prop=opt.dropprob,
            max_length=token2bert_embeddings.max_length,
            device=opt.device
        )

    # Initialize BertWCEClassifier
    model = BertWCEClassifier(
        net_type=net_type,
        output_size=nC,
        hidden_size=hidden,
        token2bert_embeddings=token2bert_embeddings,
        token2wce_embeddings=token2wce_embeddings
    )

    # Apply Xavier initialization if required
    model.xavier_uniform()

    # Return embedding sizes for logging
    embsizeX = token2bert_embeddings.dim()
    embsizeY = token2wce_embeddings.dim() if token2wce_embeddings else 0
    lrnsizeX, lrnsizeY = 0, 0  # No learnable embeddings for BertWCEClassifier

    return model.to(opt.device), embsizeX, embsizeY, lrnsizeX, lrnsizeY





def set_method_name():

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
        - WCE (torch.Tensor or None):
            The supervised embeddings matrix of shape (vocabsize, num_labels),
        - sup_range (list or None): 
            A list indicating the range of supervised embeddings in the final matrix, 
            or `None` if no supervised embeddings are included.
        - dynamic_vocabsize (int): Vocabulary size matching the dynamic embeddings.

    
    Notes:
    -----
    - Pretrained embeddings are extracted using the `extract` method of the `pretrained` object.
    - Supervised embeddings are calculated using `get_supervised_embeddings`.
    - Tokens not found in the pretrained vocabulary are handled as OOV.
    """

    print(f'[embedding_matrix]: pretrained={opt.pretrained}, supervised={opt.supervised}, vocabsize={vocabsize}')

    pretrained_embeddings = None
    WCE = None
    sup_range = None
    dynamic_vocabsize = vocabsize  # Default to input vocabsize unless adjusted dynamically

    if opt.pretrained or opt.supervised:
        pretrained_embeddings = []

        # Handle pretrained embeddings
        if pretrained is not None:
            if opt.pretrained in ['glove', 'word2vec', 'fasttext']:
                # Static word-based embeddings
                print("Extracting embeddings using the static method for word-based embeddings...")
                word_list = get_word_list(word2index, out_of_vocabulary)
                weights = pretrained.extract(word_list)
                pretrained_embeddings.append(weights)
                print('\t[pretrained-matrix]', weights.shape)
            else:
                raise ValueError(f"Unsupported embedding type: {opt.pretrained}")

            """
            elif opt.pretrained in ['bert', 'roberta', 'distilbert']:
                # Dynamic token-based embeddings
                print("Extracting embeddings dynamically for dataset text with attention...")
                max_length = pretrained.tokenizer.model_max_length  # Ensure proper padding
                batch_size = opt.batch_size
                all_embeddings = []
                
                # Process dataset text in batches with a progress bar
                dataset_texts = dataset.devel_raw + dataset.test_raw  # Combine development and test set

                # Initialize the progress bar
                pbar = tqdm(total=len(dataset_texts), desc="Computing dynamic embeddings from transformer model...", unit="docs")
                with torch.no_grad():
                    for i in range(0, len(dataset_texts), batch_size):
                        batch_texts = dataset_texts[i:i + batch_size]
                        tokens = pretrained.tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=max_length
                        ).to(opt.device)

                        outputs = pretrained.model(
                            input_ids=tokens["input_ids"],
                            attention_mask=tokens["attention_mask"]
                        )
                        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                        all_embeddings.append(embeddings)

                        # Update progress bar
                        pbar.update(len(batch_texts))
                
                # Close the progress bar
                pbar.close()

                # Concatenate all embeddings
                all_embeddings = torch.cat(all_embeddings, dim=0)

                # Adjust `final_vocabsize` to match the number of embeddings
                dynamic_vocabsize = all_embeddings.shape[0]
                print(f"Dynamic vocabulary size computed: {dynamic_vocabsize}")

                pretrained_embeddings.append(all_embeddings)
                print('\t[pretrained-matrix]', all_embeddings.shape)
            """

        # Handle supervised embeddings (WCE)
        if opt.supervised:
            
            print("Generating supervised embeddings (WCE)...")
            Xtr, _ = dataset.vectorize()
            Ytr = dataset.devel_labelmatrix
            WCE = get_supervised_embeddings(
                Xtr,
                Ytr,
                method=opt.supervised_method,
                max_label_space=opt.max_label_space,
                dozscore=(not opt.nozscore),
                debug=True
            )
            num_missing_rows = vocabsize - WCE.shape[0]
            WCE = np.vstack((WCE, np.zeros(shape=(num_missing_rows, WCE.shape[1]))))
            WCE = torch.from_numpy(WCE).float()

            if torch.isnan(WCE).any() or torch.isinf(WCE).any():
                raise ValueError("[ERROR] WCE contains NaN or Inf values during initialization.")

            print('\t[supervised-matrix]', WCE.shape)

            # Adjust the supervised range
            offset = 0
            if pretrained_embeddings:
                offset = pretrained_embeddings[0].shape[1]
            sup_range = [offset, offset + WCE.shape[1]]
            pretrained_embeddings.append(WCE)

        # Concatenate all embeddings
        pretrained_embeddings = [emb.to(opt.device) for emb in pretrained_embeddings]
        pretrained_embeddings = torch.cat(pretrained_embeddings, dim=1)

    return pretrained_embeddings, WCE, sup_range, dynamic_vocabsize



def embedding_matrix_old(dataset, pretrained, vocabsize, word2index, out_of_vocabulary, opt):
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
        - WCE (torch.Tensor or None):
            The supervised embeddings matrix of shape (vocabsize, num_labels),
        - sup_range (list or None): 
            A list indicating the range of supervised embeddings in the final matrix, 
            or `None` if no supervised embeddings are included.

    Notes:
    -----
    - Pretrained embeddings are extracted using the `extract` method of the `pretrained` object.
    - Supervised embeddings are calculated using `get_supervised_embeddings`.
    - Tokens not found in the pretrained vocabulary are handled as OOV.
    """

    print('[embedding_matrix]')

    pretrained_embeddings = None
    WCE = None
    sup_range = None
    
    if opt.pretrained or opt.supervised:

        pretrained_embeddings = []

        if (pretrained is not None):
            word_list = get_word_list(word2index, out_of_vocabulary)
            weights = pretrained.extract(word_list)
            pretrained_embeddings.append(weights)
            print('\t[pretrained-matrix] ', weights.shape)
            del pretrained

        if opt.supervised:
            Xtr, _ = dataset.vectorize()
            Ytr = dataset.devel_labelmatrix
            WCE = get_supervised_embeddings(Xtr, Ytr,
                                          method=opt.supervised_method,
                                          max_label_space=opt.max_label_space,
                                          dozscore=(not opt.nozscore))
            num_missing_rows = vocabsize - WCE.shape[0]
            WCE = np.vstack((WCE, np.zeros(shape=(num_missing_rows, WCE.shape[1]))))
            WCE = torch.from_numpy(WCE).float()
            print('\t[supervised-matrix]', WCE.shape)

            offset = 0
            if pretrained_embeddings:
                offset = pretrained_embeddings[0].shape[1]
            sup_range = [offset, offset + WCE.shape[1]]
            pretrained_embeddings.append(WCE)
        
        pretrained_embeddings = [emb.to(opt.device) for emb in pretrained_embeddings]
        pretrained_embeddings = torch.cat(pretrained_embeddings, dim=1)

    return pretrained_embeddings, WCE, sup_range


def init_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

def init_loss(classification_type):
    assert classification_type in ['multilabel','multi-label','singlelabel', 'single-label'], 'unknown classification mode'
    L = torch.nn.BCEWithLogitsLoss() if classification_type in ['multilabel', 'multi-label'] else torch.nn.CrossEntropyLoss()
    return L.cuda()


"""
def init_logfile(method_name, opt):
    logfile = CSVLog(opt.log_file, ['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'])
    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', method_name)
    assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'
    return logfile
"""


def load_pt_model(opt):

    print(f'loading pretrained model... opt.pretrained: {opt.pretrained}')

    if opt.pretrained == 'glove':
        return GloVeEmbeddings(setname=GLOVE_SET, model_name=GLOVE_MODEL, path=opt.glove_path)
    elif opt.pretrained == 'word2vec':
        return Word2VecEmbeddings(path=opt.word2vec_path+WORD2VEC_MODEL, limit=1000000)
    elif opt.pretrained == 'fasttext':
        return FastTextEmbeddings(path=opt.fasttext_path+FASTTEXT_MODEL, limit=1000000)
    else:
        raise ValueError(f"Unsupported pretrained model type: {opt.pretrained}")

    """
    elif opt.pretrained == 'bert':
        return BERTEmbeddings(device=opt.device, batch_size=opt.batch_size, path=opt.bert_path)
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
    """




def train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name):

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
                ytr = vstack((ytr[from_:], ytr[:to_]))
            else:
                ytr = np.concatenate((ytr[from_:], ytr[:to_]))
        else:
            train_index = train_index[from_:to_]
            ytr = ytr[from_:to_]

    model.train()
    for idx, (batch, target) in enumerate(batchify(train_index, ytr, opt.batch_size, pad_index, opt.device, as_long)):
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
            batchify(test_index, yte, opt.batch_size_test, pad_index, opt.device, target_long=target_long),
            desc='evaluation: '
    ):
        logits = model(batch)
        loss = criterion(logits, target).item()
        prediction = csr_matrix(predict(logits, classification_type=classification_type))
        predictions.append(prediction)

    yte_ = scipy.sparse.vstack(predictions)
    
    Mf1, mf1, acc = evaluation_legacy(yte, yte_, classification_type)
    print(f'[{measure_prefix}] Macro-F1={Mf1:.3f} Micro-F1={mf1:.3f} Accuracy={acc:.3f}')
    
    tend = time() - tinit

    if (measure_prefix == 'final-te'):

        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

        """
        logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
        logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
        logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
        logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)
        """

    return Mf1


def main(opt):

    program = 'wrd_layer_cake'
    version = '2.0'

    print(f'\n\t--- WORD_LAYER_CAKE Version: {version} ---')
    print()

    # initialize logging and other system run variables
    already_modelled, logfile, method_name, pretrained, embeddings, embedding_type, emb_path, lm_type, mode, system = initialize_testing(opt, program=program, version=version)

    print("method_name:", method_name)
    print("pretrained:", pretrained)
    print("embeddings:", embeddings)    
    print("embedding_type:", embedding_type)
    print("emb_path:", emb_path)
    print("lm_type:", lm_type)
    print("mode:", mode)
    print("system:", system)

    # check to see if model params have been computed already
    if (already_modelled and not opt.force):
        print(f'--- model {method_name} with embeddings {embeddings}, pretrained == {pretrained}, tunable == {opt.tunable}, and wc_supervised == {opt.supervised} for {opt.dataset} already calculated, run with --force option to override. ---')
        exit(0)

    pt_model = load_pt_model(opt)
    print("pt_model:", pt_model)

    dataset = Dataset.load(
        name=opt.dataset, 
        vtype=opt.vtype,
        pt_model=pt_model,
        pickle_dir=opt.pickle_dir 
    ).show()

    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, opt, pt_model=pt_model)
    print("word2index:", type(word2index), len(word2index))
    print("out_of_vocabulary:", type(out_of_vocabulary), len(out_of_vocabulary))
    vocabsize = len(word2index) + len(out_of_vocabulary)
    print("vocabsize:", vocabsize)

    # dataset split tr/val/test
    val_size = min(int(len(devel_index) * .2), 20000)
    train_index, val_index, ytr, yval = train_test_split(
        devel_index, dataset.devel_target, test_size=val_size, random_state=opt.seed, shuffle=True
    )
    yte = dataset.test_target

    pretrained_embeddings, WCE, sup_range, vocabsize = embedding_matrix(dataset, pt_model, vocabsize, word2index, out_of_vocabulary, opt)

    print("pretrained_embeddings:", type(pretrained_embeddings), pretrained_embeddings.shape)
    if (WCE is not None):
        print("WCE:", type(WCE), WCE.shape)
    else:
        print("WCE: None")
    print("sup_range:", sup_range)
    print("vocabsize:", vocabsize)

    # Initialize the model
    lc_model, embedding_sizeX, embedding_sizeY, lrn_sizeX, lrn_sizeY = init_Net(
        dataset.nC, vocabsize, pretrained_embeddings, sup_range, opt
        )
    print("lc_model:\n", lc_model)

    optim = init_optimizer(lc_model, lr=opt.lr, weight_decay=opt.weight_decay)
    print("optim:\n", optim)
    
    criterion = init_loss(dataset.classification_type)
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
    early_stop = EarlyStopping(lc_model, patience=opt.patience, checkpoint=f'{opt.checkpoint_dir}/{opt.net}-{opt.dataset}')

    for epoch in range(1, opt.nepochs + 1):
        train(lc_model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name)

        # validation
        macrof1 = test(lc_model, val_index, yval, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'va', opt, embedding_size=emb_size_str)
        early_stop(macrof1, epoch)
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

        if opt.val_epochs>0:
            print(f'last {opt.val_epochs} epochs on the validation set')
            for val_epoch in range(1, opt.val_epochs + 1):
                train(res_model, val_index, yval, pad_index, tinit, logfile, criterion, optim, epoch+val_epoch, method_name)

        # test
        print('Training complete: testing')
        test(res_model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'final-te', opt, embedding_size=emb_size_str)




if __name__ == '__main__':

    available_datasets = Dataset.dataset_available
    available_dropouts = {'sup','none','full','learn'}

    # Training settings
    parser = argparse.ArgumentParser(description='Neural text classification with Word-Class Embeddings')
    parser.add_argument('--dataset', type=str, default='reuters21578', metavar='str',
                        help=f'dataset, one in {available_datasets}')
    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', 
                        help=f'dataset base vectorization strategy, in [tfidf, count]')
    parser.add_argument('--batch-size', type=int, default=100, metavar='int',
                        help='input batch size (default: 100)')
    parser.add_argument('--batch-size-test', type=int, default=250, metavar='int',
                        help='batch size for testing (default: 250)')
    parser.add_argument('--nepochs', type=int, default=200, metavar='int',
                        help='number of epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=10, metavar='int',
                        help='patience for early-stop (default: 10)')
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
    parser.add_argument('--dropprob', type=float, default=0.5, metavar='[0.0, 1.0]',
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='int',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='int',
                        help='how many batches to wait before printing training status')
    parser.add_argument('--log-file', type=str, default='../log/log.csv', metavar='str',
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
    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext',
                        help='pretrained embeddings, use "glove", "word2vec", or "fasttext" (default None)')
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use supervised embeddings')
    parser.add_argument('--sup-mode', type=str, default='cat', help='How WCEs are combined with model embeddings (cat)')
    parser.add_argument('--supervised-method', type=str, default='ig', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    parser.add_argument('--learnable', type=int, default=0, metavar='int',
                        help='dimension of the learnable embeddings (default 0)')
    parser.add_argument('--val-epochs', type=int, default=1, metavar='int',
                        help='number of training epochs to perform on the validation set once training is '
                             'over (default 1)')
    
    parser.add_argument('--glove-path', type=str, default=VECTOR_CACHE+'/GloVe/',
                        metavar='PATH',
                        help=f'path to glove.840B.300d pretrained vectors (used only with --pretrained glove)')
    parser.add_argument('--word2vec-path', type=str, default=VECTOR_CACHE+'/Word2Vec/',
                        metavar='PATH',
                        help=f'path to GoogleNews-vectors-negative300.bin pretrained vectors (used only '
                             f'with --pretrained word2vec)')
    parser.add_argument('--fasttext-path', type=str, default=VECTOR_CACHE+'/fastText/',
                        metavar='PATH',
                        help=f'path to crawl-300d-2M.vec pretrained vectors (used only with --pretrained fasttext)')
    parser.add_argument('--bert-path', type=str, default=VECTOR_CACHE+'/BERT',
                        metavar='PATH',
                        help=f'Directory to BERT pretrained vectors, defaults to {VECTOR_CACHE}/BERT. Used only with --pretrained bert')
    
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
        opt.batch_size = DEFAULT_GPU_BATCH_SIZE
    elif torch.backends.mps.is_available():
        opt.device = torch.device("mps")
        opt.batch_size = DEFAULT_MPS_BATCH_SIZE
    else:
        opt.device = torch.device("cpu")
        opt.batch_size = DEFAULT_CPU_BATCH_SIZE
    print(f'running on {opt.device}')
    print("batch_size:", opt.batch_size)

    torch.manual_seed(opt.seed)

    assert opt.dataset in available_datasets, \
        f'unknown dataset {opt.dataset}'
    assert opt.pretrained in [None]+AVAILABLE_PRETRAINED, \
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

    if opt.pickle_dir:
        opt.pickle_path = join(opt.pickle_dir, f'{opt.dataset}.pickle')

    main(opt)