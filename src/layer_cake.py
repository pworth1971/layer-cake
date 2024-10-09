import argparse
from time import time
import logging

import scipy
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split

# custom classes 
from embedding.pretrained import *
from embedding.supervised import *

from util.metrics import *
from data.tsr_function__ import *

from data.lc_dataset import LCDataset, save_to_pickle, load_from_pickle, loadpt_data

from model.classification import ml_classification

from util.csv_log import CSVLog

from model.LCRepresentationModel import FASTTEXT_MODEL, GLOVE_MODEL, WORD2VEC_MODEL
from model.LCRepresentationModel import BERT_MODEL, ROBERTA_MODEL, XLNET_MODEL, GPT2_MODEL



from data.lc_dataset import LCDataset, save_to_pickle, load_from_pickle


from model.LCRepresentationModel import FASTTEXT_MODEL, GLOVE_MODEL, WORD2VEC_MODEL
from model.LCRepresentationModel import BERT_MODEL, ROBERTA_MODEL, XLNET_MODEL, GPT2_MODEL


from model.classification import NeuralClassifier
from util.early_stop import EarlyStopping
from util.file import create_if_not_exist


import argparse

import warnings
warnings.filterwarnings("ignore")


import torchtext
torchtext.disable_torchtext_deprecation_warning()



#
# TODO: Set up logging
#
logging.basicConfig(filename='../log/application.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


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


# ---------------------------------------------------------------------------------------------------------------------------

def init_Net(opt, nC, vocabsize, pretrained_embeddings, sup_range, device):
    
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

    #logging.info(f"embedding_matrix()...")
    
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


# ------------------------------------------------------------------------------------------------------------------------------------------------

def layer_cake(opt, logfile, pretrained_vectors, method_name, dataset, word2index, out_of_vocabulary, pad_index, devel_index, test_index):
    """
    #main driver method for all logic, called from the __main__ method after args are parsed
    input is a Namespace of arguments that the program was run with, i.e. opts or args, or a line 
    from a config batch file with all the arguments if being run in batch mode
    """
    print("\t-- layer_cake() -- ")

    #print("Command line:", ' '.join(sys.argv))                  # Print the full command line

    print("opt:", type(opt), opt)

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
        pretrained_vectors, 
        vocabsize, 
        word2index, 
        out_of_vocabulary, 
        opt
        )

    if (pretrained_embeddings == None):
        print('\t[pretrained_embeddings]\n\t', None)
        embedding_dims = -1
    else:
        print('\t[pretrained_embeddings]\n\t', pretrained_embeddings.shape)
        # Log the dimensions of the embeddings if available
        embedding_dims = pretrained_embeddings.shape[1]
        print(f"Number of dimensions (embedding size): {embedding_dims}")

    loss_history = {'train_loss': [], 'test_loss': []}              # Initialize loss tracking

    print("setting up model...")
    model = init_Net(opt, dataset.nC, vocabsize, pretrained_embeddings, sup_range, opt.device)
    optim = init_optimizer(model, lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = init_loss(dataset.classification_type)

    # train-validate
    tinit = time.time()
    create_if_not_exist(opt.checkpoint_dir)
    early_stop = EarlyStopping(model, patience=opt.patience, checkpoint=f'{opt.checkpoint_dir}/{opt.net}-{opt.dataset}')

    for epoch in range(1, opt.nepochs + 1):

        print(" \n-------------- EPOCH ", {epoch}, "-------------- ")    
        train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, opt.dataset, epoch, method_name, loss_history)
        
        macrof1, test_loss = test(model, val_index, yval, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'va', loss_history)

        early_stop(macrof1, epoch)

        if opt.test_each>0:
            if (opt.plotmode and (epoch==1 or epoch%opt.test_each==0)) or (not opt.plotmode and epoch%opt.test_each==0 and epoch<opt.nepochs):
                test(model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'te', loss_history)

        if early_stop.STOP:
            print('[early-stop]')
            if not opt.plotmode:                # with plotmode activated, early-stop is ignored
                break

    print("\t...restoring best model...")

    # restores the best model according to the Mf1 of the validation set (only when plotmode==False)
    stoptime = early_stop.stop_time - tinit
    stopepoch = early_stop.best_epoch

    #logfile.add_layered_row(epoch=stopepoch, measure=f'early-stop', value=early_stop.best_score, timelapse=stoptime)
    logfile.insert(dimensions=embedding_dims, epoch=stopepoch, measure=f'early-stop', value=early_stop.best_score, timelapse=stoptime)

    if not opt.plotmode:
        print()
        print('...performing final evaluation...')
        model = early_stop.restore_checkpoint()

        if opt.val_epochs>0:
            print(f'last {opt.val_epochs} epochs on the validation set')
            for val_epoch in range(1, opt.val_epochs + 1):
                train(model, val_index, yval, pad_index, tinit, logfile, criterion, optim, opt.dataset, epoch+val_epoch, method_name, loss_history)

        # test
        print('Training complete: testing')
        test_loss = test(model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'final-te', loss_history)


    if (opt.plotmode):                                          # Plot the training and testing loss after all epochs
        plot_loss_over_epochs( opt.dataset, {
            'epochs': np.arange(1, len(loss_history['train_loss']) + 1),
            'train_loss': loss_history['train_loss'],
            'test_loss': loss_history['test_loss']
        }, method_name, '../output')

# ------------------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------------------------

def train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, dataset, epoch, method_name, loss_history):
    
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

    # Initialize a variable to store the number of dimensions
    dims = None

    for idx, (batch, target) in enumerate(batchify(train_index, ytr, opt.batch_size, pad_index, opt.device, as_long)):

        if dims is None:
            dims = batch.shape[1]  # Get the number of features (columns) from the batch
            print("# dimensions:", dims)

        optim.zero_grad()
        loss = criterion(model(batch), target)
        loss.backward()
        clip_gradient(model)
        optim.step()
    
        epoch_loss += loss.item()
        total_batches += 1

        if idx % opt.log_interval == 0:
            interval_loss = loss.item()
            print(f'{dataset} {method_name} Epoch: {epoch}, Step: {idx}, Training Loss: {interval_loss:.6f}')

    mean_loss = np.mean(interval_loss)
    
    loss_history['train_loss'].append(mean_loss)

    #logfile.add_layered_row(epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time.time() - tinit)
    logfile.insert(dimnsions=dims, epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time.time() - tinit)

    return mean_loss

# ------------------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------------------------

def test(model, test_index, yte, pad_index, classification_type, tinit, epoch, logfile, criterion, measure_prefix, loss_history):
    
    print("..testing...")

    model.eval()
    predictions = []

    test_loss = 0
    total_batches = 0

    target_long = isinstance(criterion, torch.nn.CrossEntropyLoss)

    dims = -1

    # Retrieve the number of dimensions from the first example in test_index
    if isinstance(test_index, list) and len(test_index) > 0:
        dims = len(test_index[0])  # Assuming each item in test_index is a list/vector of features
        print(f"Number of dimensions in test data: {dims}")
    else:
        print("Unable to determine the number of dimensions from test_index")
    
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
    
    print("evaluating test run...")
    
    #Mf1, mf1, acc = evaluation(yte, yte_, classification_type)
    Mf1, mf1, acc, h_loss, precision, recall, j_index = evaluation_nn(yte, yte_, classification_type)
    
    print(f'[{measure_prefix}] Macro-F1={Mf1:.3f} Micro-F1={mf1:.3f} Accuracy={acc:.3f}')
    
    tend = time.time() - tinit

    """
    if classification_type == 'multilabel':
        Mf1_orig, mf1_orig, acc_orig = multilabel_eval_orig(yte, yte_)
        print("--original calc--")
        print(f'[{measure_prefix}] Macro-F1={Mf1_orig:.3f} Micro-F1={mf1_orig:.3f} Accuracy={acc_orig:.3f}')
    """

    """
    logfile.add_layered_row(epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

    logfile.add_layered_row(epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
    logfile.add_layered_row(epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)
    """

    logfile.insert(dimensions=dims, epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
    logfile.insert(dimensions=dims, epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
    logfile.insert(dimensions=dims, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    logfile.insert(dimensions=dims, epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

    logfile.insert(dimensions=dims, epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dimensions=dims, epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dimensions=dims, epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dimensions=dims, epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)
    
    mean_loss = test_loss / total_batches
    loss_history['test_loss'].append(mean_loss)

    logfile.insert(dimensions=dims, epoch=epoch, measure=f'{measure_prefix}-loss', value=mean_loss, timelapse=time.time() - tinit)

    return Mf1, mean_loss                          # Return value for use in early stopping and loss plotting

# ------------------------------------------------------------------------------------------------------------------------------------------------



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
    logger.set_default('pretrained', pretrained)
    logger.set_default('embeddings', embeddings)
    logger.set_default('run', args.seed)
    logger.set_default('tunable', args.tunable)
    logger.set_default('representation', method_name)
    logger.set_default('class_type', lm_type)

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
        embeddings=embeddings,
        model=args.net, 
        representation=method_name,
        pretrained=pretrained, 
        tunable=args.tunable,
        wc_supervised=args.supervised,
        run=args.seed
        )

    print("already_modelled:", already_modelled)

    return already_modelled, logger, method_name, pretrained, embeddings, emb_path, lm_type, mode, system

    """
    @classmethod
    def load_nn(cls, dataset_name, vectorization_type='tfidf', embedding_type='word', base_pickle_path=None):

        print("Dataset::load():", dataset_name, base_pickle_path)

        print("vectorization_type:", vectorization_type)
        print("embedding_type:", embedding_type)

        # Create a pickle path that includes the vectorization type
        # NB we assume the /pickles directory exists already
        if base_pickle_path:
            full_pickle_path = f"{base_pickle_path}{'/'}{dataset_name}_{vectorization_type}.pkl"
            pickle_file_name = f"{dataset_name}_{vectorization_type}.pkl"
        else:
            full_pickle_path = None
            pickle_file_name = None

        print("pickle_file_name:", pickle_file_name)

        # not None so we are going to create the pickle file, 
        # by dataset and vectorization type
        if full_pickle_path:
            print("full_pickle_path: ", {full_pickle_path})

            if os.path.exists(full_pickle_path):                                        # pickle file exists, load it
                print(f'loading pickled dataset from {full_pickle_path}')
                dataset = pickle.load(open(full_pickle_path, 'rb'))
            else:                                                                       # pickle file does not exist, create it, load it, and dump it
                print(f'fetching dataset and dumping it into {full_pickle_path}')
                dataset = LCDataset(name=dataset_name, vectorization_type=vectorization_type, embedding_type=embedding_type)

                print('dumping')
                #pickle.dump(dataset, open(pickle_path, 'wb', pickle.HIGHEST_PROTOCOL))
                # Open the file for writing and write the pickle data
                try:
                    with open(full_pickle_path, 'wb', pickle.HIGHEST_PROTOCOL) as file:
                        pickle.dump(dataset, file)
                    print("data successfully pickled at:", full_pickle_path)
                except Exception as e:
                    print(f'\n\t------*** ERROR: Exception raised, failed to pickle data: {e} ***------')

        else:
            print(f'loading dataset {dataset_name}')
            dataset = LCDataset(name=dataset_name, vectorization_type=vectorization_type, embedding_type=embedding_type)

        return dataset

    """
    

# --------------------------------------------------------------------------------------------------------------------------------------
#
# command line argument, program: parser plus assertions + main(opt)
#
# --------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    available_datasets = LCDataset.dataset_available
    available_dropouts = {'sup','none','full','learn'}

    
    print("\n\t------------------------------------- LAYER CAKE: Neural text classification with Word-Class Embeddings -------------------------------------\n")

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
    
    parser.add_argument('--log-file', type=str, default='../log/lc_nn_test.test', metavar='str',
                        help='path to the log logger output file')
    
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str',
                        help=f'if set, specifies the path where to save/load the dataset pickled (set to None if you '
                             f'prefer not to retain the pickle file)')
    
    parser.add_argument('--test-each', type=int, default=0, metavar='int',
                        help='how many epochs to wait before invoking test (default: 0, only at the end)')
    
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoint', metavar='str',
                        help='path to the directory containing checkpoints')
    
    parser.add_argument('--net', type=str, default='lstm', metavar='str',
                        help=f'net, one in {NeuralClassifier.ALLOWED_NETS}')
    
    parser.add_argument('--pretrained', type=str, default=None, metavar='embeddings',
                        help='pretrained embeddings, use "glove", "word2vec", "fasttext", "bert", "roberts", "xlnet", "gpt2", or "llama" (default None)')
    
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use supervised embeddings')
    
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    
    parser.add_argument('--learnable', type=int, default=0, metavar='int',
                        help='dimension of the learnable embeddings (default 0)')
    
    parser.add_argument('--val-epochs', type=int, default=1, metavar='int',
                        help='number of training epochs to perform on the validation set once training is over (default 1)')
    
    parser.add_argument('--glove-path', type=str, default=VECTOR_CACHE+'/GloVe', metavar='PATH',
                        help=f'Drectory to pretrained GloVe embeddings, defaults to {VECTOR_CACHE}/GloVe. Used only with --pretrained glove, '
                            f'defaults to {VECTOR_CACHE}.') 

    parser.add_argument('--word2vec-path', type=str, default=VECTOR_CACHE+'/Word2Vec', metavar='PATH',
                        help=f'Directory to Word2Vec pretrained vectors (e.g. GoogleNews-vectors-negative300.bin), used only '
                             f'with --pretrained word2vec. Defaults to {VECTOR_CACHE}/Word2Vec.')
    
    parser.add_argument('--fasttext-path', type=str, default=VECTOR_CACHE+'/fastText', metavar='PATH',
                        help=f'Directory to fastText pretrained vectors, defaults to {VECTOR_CACHE}/fastText, used only with --pretrained fasttext')
    
    parser.add_argument('--bert-path', type=str, default=VECTOR_CACHE+'/BERT',
                        metavar='PATH',
                        help=f'Directory to BERT pretrained vectors, defaults to {VECTOR_CACHE}/BERT. Used only with --pretrained bert')

    parser.add_argument('--roberta-path', type=str, default=VECTOR_CACHE+'/RoBERTa',
                        metavar='PATH',
                        help=f'Directory to RoBERTa pretrained vectors, defaults to {VECTOR_CACHE}/RoBERTA. Used only with --pretrained roberta')
    
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
    
    parser.add_argument('--batch-file', type=str, default=None, metavar='str',
                        help='path to the config file used for batch processing of multiple experiments')

    opt = parser.parse_args()
    print("opt:", type(opt), opt)

    # Setup device prioritizing CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        opt.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        opt.device = torch.device("mps")
    else:
        opt.device = torch.device("cpu")
        
    print(f'running on {opt.device}')

    torch.manual_seed(opt.seed)

    # single run case
    if (opt.batch_file is None):                        # running single command 

        print("single run processing...")

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
        
        # initialize logging and other system run variables
        already_modelled, logfile, method_name, pretrained, embeddings, emb_path, lm_type, mode, system = initialize_testing(opt)

        # check to see if model params have been computed already
        if (already_modelled and not opt.force):
            print(f'--- model {method_name} with embeddings {embeddings}, pretrained == {pretrained}, tunable == {opt.tunable}, and wc_supervised == {opt.supervised} for {opt.dataset} already calculated, run with --force option to override. ---')
            exit(0)
    
        #pretrained, pretrained_vectors = load_pretrained_embeddings(opt.pretrained, opt)

        embedding_type = get_embedding_type(opt)
        print("embedding_type:", embedding_type)

        """
        print(f"initializing dataset: {opt.dataset}")
        lcd = LCDataset.load_nn(
            dataset_name=opt.dataset,                       # dataset name
            vectorization_type=opt.vtype,                   # vectorization type 
            embedding_type=embedding_type,                  # embedding type ('word or 'token')
            base_pickle_path=opt.pickle_dir                 # base pickle path
        ).show()
        """
    
        print("embeddings:", embeddings)    
        print("embedding_path:", emb_path)
        print("embedding_type:", embedding_type)

        #
        # Load the dataset and the associated (pretrained) embedding structures
        # to be fed into the model
        #                                                          
        """
        Xtr_raw, Xte_raw, Xtr_vectorized, Xte_vectorized, y_train_sparse, y_test_sparse, target_names, class_type, embedding_vocab_matrix, Xtr_weighted_embeddings, \
            Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, \
                Xte_summary_embeddings = loadpt_data(
                                                dataset=opt.dataset,                            # Dataset name
                                                vtype=opt.vtype,                                # Vectorization type
                                                pretrained=opt.pretrained,                      # pretrained embeddings type
                                                embedding_path=emb_path,                        # path to pretrained embeddings
                                                emb_type=embedding_type                         # embedding type (word or token)
                                                )                                                
    
        #print("Xtr_raw:", type(Xtr_raw), Xtr_raw.shape)
        #print("Xte_raw:", type(Xte_raw), Xte_raw.shape)

        print("Xtr_vectorized:", type(Xtr_vectorized), Xtr_vectorized.shape)
        print("Xte_vectorized:", type(Xte_vectorized), Xte_vectorized.shape)

        print("y_train_sparse:", type(y_train_sparse), y_train_sparse.shape)
        print("y_test_sparse:", type(y_test_sparse), y_test_sparse.shape)
        
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

        print("Xtr_weighted_embeddings:", type(Xtr_weighted_embeddings), Xtr_weighted_embeddings.shape)
        print("Xte_weighted_embeddings:", type(Xte_weighted_embeddings), Xte_weighted_embeddings.shape)
        
        print("Xtr_avg_embeddings:", type(Xtr_avg_embeddings), Xtr_avg_embeddings.shape)
        print("Xte_avg_embeddings:", type(Xte_avg_embeddings), Xte_avg_embeddings.shape)
        
        print("Xtr_summary_embeddings:", type(Xtr_summary_embeddings), Xtr_summary_embeddings.shape)
        print("Xte_summary_embeddings:", type(Xte_summary_embeddings), Xte_summary_embeddings.shape)

        if class_type in ['multilabel', 'multi-label']:

            print("multi-label case, expanding (todense) y...")

            if isinstance(y_train_sparse, (csr_matrix, csc_matrix)):
                y_train = y_train_sparse.toarray()                      # Convert sparse matrix to dense array for multi-label tasks
            if isinstance(y_test_sparse, (csr_matrix, csc_matrix)):
                y_test = y_test_sparse.toarray()                        # Convert sparse matrix to dense array for multi-label tasks
            
            print("y_train after transformation:", type(y_train), y_train.shape)
            print("y_test after transformation:", type(y_test), y_test.shape)
        else:
            y_train = y_train_sparse
            y_test = y_test_sparse
        """

        lcd = loadpt_data(
            dataset=opt.dataset,                            # Dataset name
            vtype=opt.vtype,                                # Vectorization type
            pretrained=opt.pretrained,                      # pretrained embeddings type
            embedding_path=emb_path,                        # path to pretrained embeddings
            emb_type=embedding_type                         # embedding type (word or token)
            )                                                

        print("loaded LCDataset object:", type(lcd))
        print("lcd:", lcd.show())

        pretrained_vectors = lcd.lcr_model
        pretrained_vectors.show()

        """
        #pretrained, pretrained_vectors = load_pretrained_embeddings(opt.pretrained, opt)
        #word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(lcd, pretrained_vectors)
        """

        word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(lcd, pretrained_vectors)

        layer_cake(
            opt, 
            logfile,
            pretrained_vectors, 
            method_name,
            lcd, 
            word2index, 
            out_of_vocabulary, 
            pad_index, 
            devel_index, 
            test_index)

    else: 

        #
        # we are in batch mode so we build the array of args Namespace 
        # arguments for all of the batch runs from config file
        #
        print("batch processing, reading from file:", opt.batch_file)

        #args = vars(parser.parse_args([]))  # Get default values as a dictionary

        #all_args = parse_arguments(opt.batch_file)

        #parse_arguments(opt.batch_file, args)

        """
        for opt in all_args:
            print("opt:", type(opt), opt)
        """

        # get batch config params
        configurations = parse_config_file(opt.batch_file, parser)

        line_num = 1

        last_config = None
        pretrained = False
        pretrained_vector = None
        dataset = None

        for current_config in configurations:

            # roll these two argument params over to all current_configs
            current_config.device = opt.device
            current_config.batch_file = opt.batch_file

            print(f'\n\t--------------------- processing batch configuration file line #: {line_num} ---------------------')

            print(f'current_config: {type(current_config)}: {current_config}')
            print(f'last_config: {type(last_config)}: {last_config}')

            # -------------------------------------------------------------------------------------------------------------
            # check argument parameters
            # -------------------------------------------------------------------------------------------------------------
            if (current_config.dataset not in available_datasets):
                print(f'unknown dataset in config file line # {line_num}', current_config['dataset'])
                line_num += 1
                continue
        
            if (current_config.pretrained not in [None]+AVAILABLE_PRETRAINED):
                print(f'unknown pretrained set in config file line # {line_num}', current_config['pretrained'])
                line_num += 1
                continue
        
            if (current_config.plotmode and current_config.test_each <= 0):
                print(f'plot mode implies --test-each>0, config file line # {line_num}')
                line_num += 1
                continue
        
            if (current_config.supervised_method not in STWFUNCTIONS):
                print(f'unknown supervised term weighting function in config file line # {line_num}, permitted values are {STWFUNCTIONS}')
                line_num += 1
                continue
        
            if (current_config.droptype not in available_dropouts):
                print(f'unknown dropout type in config file line # {line_num}, permitted values are {available_dropouts}')
                line_num += 1
                continue
        
            if (current_config.droptype == 'sup' and current_config.supervised == False):
                current_config.droptype = 'none'
                print('warning: droptype="sup" but supervised="False"; the droptype changed to "none"')
                logging.warning(f'droptype="sup" but supervised="False"; the droptype changed to "none"')
        
            if (current_config.droptype == 'learn' and current_config.learnable == 0):
                current_config.droptype = 'none'
                print('warning: droptype="learn" but learnable=0; the droptype changed to "none"')
                logging.warning(f'droptype="learn" but learnable=0; the droptype changed to "none"')

            if current_config.pickle_dir:
                current_config.pickle_path = join(current_config.pickle_dir, current_config.dataset + '.pickle')
            # -------------------------------------------------------------------------------------------------------------

            already_modelled = False

            # initialize log file
            already_modelled, logfile, method_name, cpus, mem, gpus, pretrained, embeddings_log_val = initialize_logfile(current_config)

            # check to see if model params have been computed already
            if (already_modelled) and not (current_config.force):
                print(f'Assertion warning: model {method_name} with embeddings {embeddings_log_val}, pretrained == {pretrained}, tunable == {current_config.tunable}, and wc_supervised == {current_config.supervised} for {current_config.dataset} already calculated.')
                print("Run with --force option to override, continuing...")
                line_num += 1
                continue

            # initialize embeddings if need be
            if 'pretrained' in current_config and (not last_config or current_config.pretrained != last_config.pretrained):
                print(f"loading pretrained embeddings: {current_config.pretrained}")
                pretrained, pretrained_vectors = load_pretrained_embeddings(current_config.pretrained, current_config)

            # initialize dataset if need be
            if 'dataset' in current_config and (not last_config or current_config.dataset != last_config.dataset):
                print(f"initializing dataset: {current_config.dataset}")

                dataset = LCDataset.load(dataset_name=current_config.dataset, base_pickle_path=current_config.pickle_dir).show()
                word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained_vectors)

            # run layer_cake
            layer_cake(
                current_config,                         # current_onfig is already a Namespace
                logfile, 
                pretrained_vectors, 
                method_name,
                dataset, 
                word2index, 
                out_of_vocabulary,  
                pad_index, 
                devel_index, 
                test_index)
            
            last_config = current_config  # Update last_config to current for next iteration check

            line_num += 1



    # --------------------------------------------------------------------------------------------------------------------------------------
