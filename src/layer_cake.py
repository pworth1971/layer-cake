"""
layer_cake.py:
--------------------------------------------------------------------------------------------------------------------------------
Driver code to test out pretrained embeddings (GloVe, Word2Vec, FastText, or BERT) and word-class embeddings as defined in 
the relevant paper (see README) into different neural network classifiers. 

Steps for Augmenting Data with Pretrained Embeddings

Loading Pretrained Embeddings:
The script loads pretrained embeddings based on the specified type (GloVe, Word2Vec, FastText, BERT) through the 
load_pretrained_embeddings function. This function takes the embedding type specified by the opt object. 

Combining Pretrained and Supervised Embeddings:
In the embedding_matrix function, the script combines pretrained embeddings and supervised embeddings if both are specified 
in the configuration (opt.pretrained and opt.supervised).

Constructing the Embedding Matrix:
The function embedding_matrix prepares the combined embedding matrix. This matrix is built by extracting the embeddings for the 
vocabulary of the dataset and integrating supervised embeddings if required. The script ensures that each word in the dataset's 
vocabulary has a corresponding embedding, and if not, it assigns a zero vector or a default vector for out-of-vocabulary terms.

Initializing the Neural Network:
The init_Net function receives the combined embedding matrix as an argument and uses it to initialize the NeuralClassifier model.

Integrating Embeddings into the Model:
The function init_Net initializes the neural network with the combined embedding matrix. The model uses these embeddings as 
input representations for the text data. Depending on the droptype option, the embeddings may undergo dropout to prevent overfitting.
--------------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------------------------
The main() function orchestrates the training and testing processes, ensuring the model is trained on the dataset augmented with 
the pretrained embeddings and then evaluated on the test data. pretrained_mbeddings are loaded based on the specified type and 
integrated into the model's embedding layer. supervised_embeddings, if specified in the program arguments, are also integrated into the
neural model, adding semantic information from the labels directly into the embeddings. If specified in the arguments, the final 
embedding matrix can combine both pretrained and supervised embeddings, ensuring comprehensive input representations for the neural 
network. By combining pretrained and supervised embeddings, the model leverages rich, pre-learned semantic information from large 
corpora (via pretrained embeddings) and task-specific label information (via supervised embeddings) to enhance its performance on 
specific tasks.

The pretrained embeddings are integrated into the model in the init_Net function, which initializes the neural network. The 
embedding_matrix function creates the combined embedding matrix by concatenating the pretrained and supervised embeddings (if both 
are specified). This combined matrix is then returned for use in initializing the model. 

Model Initialization:
The NeuralClassifier is initialized with various parameters, including pretrained_embeddings, which is the combined matrix of 
pretrained and possibly supervised embeddings.

Embedding Layer:
Inside the NeuralClassifier, there is likely an embedding layer that uses these pretrained embeddings as its weights. This allows 
the model to use rich, pre-learned semantic information from these embeddings.

Dropout and Fine-tuning:
Depending on the droptype and tunable options, the embeddings may be subject to dropout to prevent overfitting, and they may also be 
fine-tuned during training to better adapt to the specific task. By setting the pretrained parameter in the NeuralClassifier, the 
model incorporates these embeddings into its architecture, enhancing its ability to understand and process the input text data effectively.
--------------------------------------------------------------------------------------------------------------------------------
"""

import argparse
from time import time
import matplotlib.pyplot as plt
import os
import logging

import torchtext

import scipy
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split

# custom classes 
from embedding.supervised import get_supervised_embeddings, STWFUNCTIONS
from embedding.pretrained import *

from model.classification import NeuralClassifier

from util.early_stop import EarlyStopping
from util.common import *
from util.csv_log import CSVLog
from util.file import create_if_not_exist
from util.metrics import *

from data.dataset import *

import warnings
warnings.filterwarnings("ignore")


#
# TODO: Set up logging
#
logging.basicConfig(filename='../log/application.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')



# ---------------------------------------------------------------------------------------------------------------------------

def init_Net(nC, vocabsize, pretrained_embeddings, sup_range, device):
    
    print()
    print("------------------ init_Net() ------------------")

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

    logging.info(f"Network initialized with dropout type: {opt.droptype}, dropout range: {drop_range}, dropout probability: {opt.dropprob}")

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

    return model

# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
# 
# embedding_matrix(): construct the embedding matrix that model will use for input (text) representations. 
# 
# Pretrained Embeddings Loading: If the model configuration includes using pretrained embeddings 
# (such as GloVe, Word2Vec, FastText or [BERT]), this function extracts these embeddings based 
# on the vocabulary of thedataset. 
# 
# Supervised Embeddings Construction: If your setup includes supervised embeddings, the function 
# constructs these embeddings based on the labels associated with the data, i.e. Word-Class Embeddings,
# so as to integrate semantic information from the labels directly into the embeddings, improving 
# model performance on specific tasks.
#
# Combination of Embeddings: If both pretrained and supervised embeddings are used, this function 
# combines them into a single embedding matrix. This combination can be a simple concatenation along the 
# feature dimension.
#
# Handling Missing Embeddings: The function also handles scenarios where certain words in the dataset 
# vocabulary might not be covered by the pretrained embeddings. This often involves assigning a zero 
# vector or some other form of default vector for these out-of-vocabulary (OOV) terms.
#
# Embedding Matrix Finalization: Finally, the function prepares the combined embedding matrix to be 
# used by the neural network. This includes ensuring the matrix is of the correct size and format,
# potentially converting it to a torch.Tensor if using PyTorch, and ensuring it's ready for GPU processing 
# if necessary.
# 
# ---------------------------------------------------------------------------------------------------------------------------
def embedding_matrix(dataset, pretrained, vocabsize, word2index, out_of_vocabulary, opt):

    print('embedding_matrix()...')

    logging.info(f"embedding_matrix()...")
    
    pretrained_embeddings = None
    sup_range = None
    
    if opt.pretrained or opt.supervised:
        pretrained_embeddings = []

        if pretrained is not None:
            word_list = get_word_list(word2index, out_of_vocabulary)
            weights = pretrained.extract(word_list)
            pretrained_embeddings.append(weights)
            print('\t[pretrained-matrix]', weights.shape)
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

        pretrained_embeddings = torch.cat(pretrained_embeddings, dim=1)
        print('\t[final pretrained_embeddings]\n\t', pretrained_embeddings.shape)

    return pretrained_embeddings, sup_range


def init_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay) 


def init_loss(classification_type):
    assert classification_type in ['multilabel','singlelabel'], 'unknown classification mode'
    L = torch.nn.BCEWithLogitsLoss() if classification_type == 'multilabel' else torch.nn.CrossEntropyLoss()
    return L.cuda()


# -------------------------------------------------------------------------------------------------------------------
#
# main function, called from coommand line with opts
#
# -------------------------------------------------------------------------------------------------------------------
def main(opt):
    
    print("\n------------------------------------------------------------------------------------------------------------------------------ layer_cake::main(opt) ------------------------------------------------------------------------------------------------------------------------------")

    print("Command line:", ' '.join(sys.argv))                  # Print the full command line

    # get system info to be used for logging below
    num_physical_cores, num_logical_cores, total_memory, avail_mem, num_cuda_devices, cuda_devices = get_sysinfo()

    cpus = f'physical:{num_physical_cores},logical:{num_logical_cores}'
    mem = total_memory
    if (num_cuda_devices >0):
        #gpus = f'{num_cuda_devices}:type:{cuda_devices[0]}'
        gpus = f'{num_cuda_devices}:{cuda_devices[0]}'
    else:
        gpus = 'None'

    method_name = set_method_name(opt)
    print("method_name:", method_name)

    if opt.pretrained:
        pretrained = True
        embeddings_log_val = opt.pretrained
    else:
        pretrained = False
        embeddings_log_val ='none'
    
    # initialize layered log file with core settings
    logfile = init_layered_logfile(
        method_name, 
        pretrained, 
        embeddings_log_val, 
        opt, 
        cpus, 
        mem, 
        gpus)    

    #assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'

    # check to see if the model has been run before
    already_modelled = logfile.already_calculated(
        dataset=opt.dataset,
        embeddings=embeddings_log_val,
        model=opt.net, 
        params=method_name,
        pretrained=pretrained, 
        tunable=opt.tunable,
        wc_supervised=opt.supervised
        )

    print("already_modelled:", already_modelled)

    if (already_modelled) and not (opt.force):
        print(f'Assertion warning: model {method_name} with embeddings {embeddings_log_val}, pretrained == {pretrained}, tunable == {opt.tunable}, and wc_supervised == {opt.supervised} for {opt.dataset} already calculated.')
        print("Run with --force option to override, exiting...")
        return

    print("loading pretrained embeddings...")
    pretrained, pretrained_vector = load_pretrained_embeddings(opt.pretrained, opt)

    print("loading dataset", {opt.dataset})
    dataset = Dataset.load(dataset_name=opt.dataset, base_pickle_path=opt.pickle_dir).show()
    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained_vector)

    print("train / test data split...")
    val_size = min(int(len(devel_index) * .2), 20000)                   # dataset split tr/val/test
    train_index, val_index, ytr, yval = train_test_split(
        devel_index, dataset.devel_target, test_size=val_size, random_state=opt.seed, shuffle=True
    )
    yte = dataset.test_target

    vocabsize = len(word2index) + len(out_of_vocabulary)
    print("vocabsize:", {vocabsize})

    # build the word embeddings based upon opt
    print("building the embeddings...")
    pretrained_embeddings, sup_range = embedding_matrix(
        dataset, 
        pretrained_vector, 
        vocabsize, 
        word2index, 
        out_of_vocabulary, 
        opt
        )

    if (pretrained_embeddings == None):
        print('\t[pretrained_embeddings]\n\t', None)
    else:
        print('\t[pretrained_embeddings]\n\t', pretrained_embeddings.shape)

    loss_history = {'train_loss': [], 'test_loss': []}              # Initialize loss tracking

    print("setting up model...")
    model = init_Net(dataset.nC, vocabsize, pretrained_embeddings, sup_range, opt.device)
    optim = init_optimizer(model, lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = init_loss(dataset.classification_type)

    # train-validate
    tinit = time()
    create_if_not_exist(opt.checkpoint_dir)
    early_stop = EarlyStopping(model, patience=opt.patience, checkpoint=f'{opt.checkpoint_dir}/{opt.net}-{opt.dataset}')

    for epoch in range(1, opt.nepochs + 1):

        print()
        print(" -------------- EPOCH ", {epoch}, "-------------- ")    
        train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name, loss_history)
        
        macrof1, test_loss = test(model, val_index, yval, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'va', loss_history)

        early_stop(macrof1, epoch)

        if opt.test_each>0:
            if (opt.plotmode and (epoch==1 or epoch%opt.test_each==0)) or (not opt.plotmode and epoch%opt.test_each==0 and epoch<opt.nepochs):
                test(model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'te', loss_history)

        if early_stop.STOP:
            print('[early-stop]')
            if not opt.plotmode:                # with plotmode activated, early-stop is ignored
                break

    print()
    print("\t...restoring best model...")
    print()

    # restores the best model according to the Mf1 of the validation set (only when plotmode==False)
    stoptime = early_stop.stop_time - tinit
    stopepoch = early_stop.best_epoch

    logfile.add_layered_row(
        epoch=stopepoch, 
        measure=f'early-stop', 
        value=early_stop.best_score, 
        timelapse=stoptime)

    if not opt.plotmode:
        print()
        print('...performing final evaluation...')
        model = early_stop.restore_checkpoint()

        if opt.val_epochs>0:
            print(f'last {opt.val_epochs} epochs on the validation set')
            for val_epoch in range(1, opt.val_epochs + 1):
                train(model, val_index, yval, pad_index, tinit, logfile, criterion, optim, epoch+val_epoch, method_name, loss_history)

        # test
        print('Training complete: testing')
        test_loss = test(model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'final-te', loss_history)


    if (opt.plotmode):                                          # Plot the training and testing loss after all epochs
        plot_loss_over_epochs( opt.dataset, {
            'epochs': np.arange(1, len(loss_history['train_loss']) + 1),
            'train_loss': loss_history['train_loss'],
            'test_loss': loss_history['test_loss']
        }, method_name, '../output')

# end main() ----------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# train()
#
# --------------------------------------------------------------------------------------------------------------------------------------

def train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name, loss_history):
    
    print("... training...")

    epoch_loss = 0
    total_batches = 0
    
    as_long = isinstance(criterion, torch.nn.CrossEntropyLoss)
    
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
    
        epoch_loss += loss.item()
        total_batches += 1

        if idx % opt.log_interval == 0:
            interval_loss = loss.item()
            print(f'{opt.dataset} {method_name} Epoch: {epoch}, Step: {idx}, Training Loss: {interval_loss:.6f}')

    mean_loss = np.mean(interval_loss)
    
    loss_history['train_loss'].append(mean_loss)

    logfile.add_layered_row(epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time() - tinit)

    return mean_loss

# end train() --------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------
# 
# test()
#
# --------------------------------------------------------------------------------------------------------------------------------------
#  
def test(model, test_index, yte, pad_index, classification_type, tinit, epoch, logfile, criterion, measure_prefix, loss_history):
    
    print()
    print("..testing...")

    model.eval()
    predictions = []

    test_loss = 0
    total_batches = 0

    target_long = isinstance(criterion, torch.nn.CrossEntropyLoss)
    
    for batch, target in tqdm(
            batchify(test_index, yte, opt.batch_size_test, pad_index, opt.device, target_long=target_long),
            desc='evaluation: '
    ):
        logits = model(batch)
        loss = criterion(logits, target).item()
        prediction = csr_matrix(predict(logits, classification_type=classification_type))
        predictions.append(prediction)

        test_loss += loss
        total_batches += 1

    yte_ = scipy.sparse.vstack(predictions)
    #Mf1, mf1, acc = evaluation(yte, yte_, classification_type)
    Mf1, mf1, acc, h_loss, precision, recall, j_index = evaluation(yte, yte_, classification_type)
    print(f'[{measure_prefix}] Macro-F1={Mf1:.3f} Micro-F1={mf1:.3f} Accuracy={acc:.3f}')
    tend = time() - tinit

    if classification_type == 'multilabel':
        Mf1_orig, mf1_orig, acc_orig = multilabel_eval_orig(yte, yte_)
        print("--original calc--")
        print(f'[{measure_prefix}] Macro-F1={Mf1_orig:.3f} Micro-F1={mf1_orig:.3f} Accuracy={acc_orig:.3f}')
        
    logfile.add_layered_row(epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

    logfile.add_layered_row(epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)

    mean_loss = test_loss / total_batches
    loss_history['test_loss'].append(mean_loss)

    logfile.add_layered_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=mean_loss, timelapse=time() - tinit)

    return Mf1, mean_loss                          # Return value for use in early stopping and loss plotting

# end test() ---------------------------------------------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------------------------------------------------------
#
# command line argument, program: parser plus assertions + main(opt)
#
# --------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    available_datasets = Dataset.dataset_available
    available_dropouts = {'sup','none','full','learn'}

    # Training settings
    parser = argparse.ArgumentParser(description='Neural text classification with Word-Class Embeddings')
    
    parser.add_argument('--dataset', type=str, default='reuters21578', metavar='str',
                        help=f'dataset, one in {available_datasets}')
    
    parser.add_argument('--batch-size', type=int, default=100, metavar='int',
                        help='input batch size (default: 100)')
    
    parser.add_argument('--batch-size-test', type=int, default=250, metavar='int',
                        help='batch size for testing (default: 250)')
    
    parser.add_argument('--nepochs', type=int, default=100, metavar='int',
                        help='number of epochs (default: 100)')
    
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
    
    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext|bert',
                        help='pretrained embeddings, use "glove", "word2vec", "fasttext", "bert", or "llama" (default None)')
    
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use supervised embeddings')
    
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    
    parser.add_argument('--learnable', type=int, default=0, metavar='int',
                        help='dimension of the learnable embeddings (default 0)')
    
    parser.add_argument('--val-epochs', type=int, default=1, metavar='int',
                        help='number of training epochs to perform on the validation set once training is '
                             'over (default 1)')
    
    parser.add_argument('--word2vec-path', type=str, default=VECTOR_CACHE+'/GoogleNews-vectors-negative300.bin',
                        metavar='PATH',
                        help=f'path + filename to Word2Vec pretrained vectors (e.g. ../.vector_cache/GoogleNews-vectors-negative300.bin), used only '
                             f'with --pretrained word2vec')
    
    parser.add_argument('--glove-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to pretrained glove embeddings (glove.840B.300d.txt.pt file), used only with --pretrained glove') 
    
    parser.add_argument('--fasttext-path', type=str, default=VECTOR_CACHE+'/crawl-300d-2M.vec',
                        metavar='PATH',
                        help=f'path + filename to fastText pretrained vectors (e.g. --fasttext-path ../.vector_cache/crawl-300d-2M.vec), used only '
                            f'with --pretrained fasttext')
    
    parser.add_argument('--bert-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to BERT pretrained vectors (e.g. bert-base-uncased-20newsgroups.pkl), used only with --pretrained bert')

    parser.add_argument('--llama-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to LLaMA pretrained vectors, used only with --pretrained llama')

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

    opt.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'RUNNING ON DEVICE == {opt.device}')

    assert f'{opt.device}'=='cuda', 'forced cuda device but cpu found'
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
        logging.warning(f'droptype="sup" but supervised="False"; the droptype changed to "none"')
    
    if opt.droptype == 'learn' and opt.learnable==0:
        opt.droptype = 'none'
        print('warning: droptype="learn" but learnable=0; the droptype changed to "none"')
        logging.warning(f'droptype="learn" but learnable=0; the droptype changed to "none"')
    
    if opt.pickle_dir:
        opt.pickle_path = join(opt.pickle_dir, f'{opt.dataset}.pickle')

    main(opt)

    # --------------------------------------------------------------------------------------------------------------------------------------
