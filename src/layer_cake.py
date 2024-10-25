import argparse
from time import time
import logging

import scipy
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# custom classes 
from embedding.pretrained import *
from embedding.supervised import *

from util.metrics import *
from data.tsr_function__ import *

from data.lc_dataset import LCDataset, loadpt_data

from util.csv_log import CSVLog

from model.LCRepresentationModel import FASTTEXT_MODEL, GLOVE_MODEL, WORD2VEC_MODEL
from model.LCRepresentationModel import BERT_MODEL, ROBERTA_MODEL, XLNET_MODEL, GPT2_MODEL

from model.classification import NeuralClassifier
from util.early_stop import EarlyStopping
from util.file import create_if_not_exist

import argparse

import warnings
warnings.filterwarnings("ignore")


"""
import torchtext
torchtext.disable_torchtext_deprecation_warning()
"""



#
# TODO: Set up logging
#
logging.basicConfig(filename='../log/application.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


"""
layer_cake.py:
--------------------------------------------------------------------------------------------------------------------------------
Driver code to test out pretrained embeddings (GloVe, Word2Vec, FastText, BERT, RoBERTa, XLNet, GPT2, or LlaMa) and word-class embeddings 
as defined in the relevant paper (see README) into different neural network classifiers. 


1. Model Structure and Embedding Types
The python code first loads pretrained embeddings based on the specified type (GloVe, Word2Vec, FastText, BERT, RoBERTA, XLNet, GPT2, 
LlaMa et al) through the load_pretrained_embeddings function. This function takes the embedding type specified by the opt object. Each 
of these embeddings is loaded based on the type specified in the command-line arguments (opt.pretrained).

Supervised Embeddings: Word-Class Embeddings (WCE) that integrate label-based semantic information directly into the model’s embeddings. 
These embeddings are task-specific and are combined with pretrained embeddings for enhanced performance. [See 2019 paper for detail]

Constructing the Embedding Matrix:
The function embedding_matrix prepares the combined embedding matrix. This matrix is built by extracting the embeddings for the 
vocabulary of the dataset and integrating supervised embeddings if required. The script ensures that each word in the dataset's 
vocabulary has a corresponding embedding, and if not, it assigns a zero vector or a default vector for out-of-vocabulary terms.

Combining Pretrained and Supervised Embeddings:
In the embedding_matrix function, the script combines pretrained embeddings and supervised embeddings if both are specified 
in the configuration (opt.pretrained and opt.supervised).


2. Data Population and Handling
Loading Pretrained Embeddings:
The function load_pretrained_embeddings retrieves embeddings based on the type specified by the user. It checks the vocabulary and 
extracts corresponding vectors.

Combining Pretrained and Supervised Embeddings:
The embedding_matrix function constructs the embedding matrix using both pretrained and supervised embeddings if both are specified.
If a word in the dataset's vocabulary is not covered by pretrained embeddings, the code handles these out-of-vocabulary terms by 
assigning a zero vector or a default fallback vector.

Dataset 
from sklearn.preprocessing import LabelEncoder

# Assuming this is defined globally or passed to the function
label_encoder = LabelEncoder()


def encode_labels(labels):
    # Fit and transform labels using the encoder if it's not already fitted
    return label_encoder.transform(labels)

# Fit the LabelEncoder once on the entire labels if not already fitted
    if not hasattr(label_encoder, 'classes_'):
        label_encoder.fit(labels)
        print("Label classes:", label_encoder.classes_)
:
The dataset is vectorized, and vocabulary indices (word2index) are created. Out-of-vocabulary (OOV) terms are managed using an OOV index. The 
train-test split is performed with the help of train_test_split from sklearn.


3. Neural Model (Classifier) Architecture 
The init_Net function receives the combined embedding matrix as an argument and uses it to initialize the NeuralClassifier model.
The initialization function sets up the NeuralClassifier, which is the main model structure. The model can be of various types, such as 
CNN, LSTM, or other custom architectures. The embeddings, dropout parameters, and other network configurations like hidden units or 
channels (for CNN) are passed as arguments. The function also includes an option for fine-tuning the embeddings (opt.tunable), allowing 
the model to adjust the pretrained embeddings to the specific task.

The pretrained embeddings are integrated into the model in the init_Net function, which initializes the neural network. The 
embedding_matrix function creates the combined embedding matrix by concatenating the pretrained and supervised embeddings (if both 
are specified). This combined matrix is then returned for use in initializing the model. 

Model Initialization:
The NeuralClassifier is initialized with various parameters, including pretrained_embeddings, which is the combined matrix of 
pretrained and possibly supervised embeddings. The neural network is initialized with the combined embedding matrix. The model 
uses these embeddings as input representations for the text data. Depending on the droptype option, the embeddings may 
undergo dropout to prevent overfitting. Inside the NeuralClassifier, there is an embedding layer that uses these pretrained 
embeddings as its weights. This allows the model to use rich, pre-learned semantic information from these embeddings.

Dropout and Fine-tuning:
Depending on the droptype and tunable options, the embeddings may be subject to dropout to prevent overfitting, and they may also be 
fine-tuned during training to better adapt to the specific task. By setting the pretrained parameter in the NeuralClassifier, the 
model incorporates these embeddings into its architecture, enhancing its ability to understand and process the input text data effectively.


4. Training Process
Training Loop:
The train function performs the training, including batching and optimization using Adam (init_optimizer). It iterates over the training 
data, computes the loss, and updates the model parameters using backpropagation. Dropout is applied based on user-specified options 
(opt.droptype), helping to prevent overfitting.

Early Stopping:
The code integrates early stopping using the EarlyStopping utility, which monitors validation performance and halts training if 
there’s no improvement.

Loss Functions:
The type of loss (Binary Cross-Entropy or Cross-Entropy) is chosen based on whether the classification task is multilabel or single-label.


5. Testing and Evaluation
The test function evaluates the model on the test data and computes metrics such as macro-F1, micro-F1, and accuracy. These metrics
provide insight into how well the model performs on the evaluation set. The code logs these metrics using the CSVLog utility, which 
tracks progress over epochs.


6. Optimization Techniques
Dropout and Fine-Tuning:
Dropout is applied to different parts of the embedding layer based on user settings, preventing overfitting. Fine-tuning 
allows the model to adapt pretrained embeddings to the specific task.

Gradient Clipping:
The code includes clip_gradient in the train function to prevent exploding gradients, particularly useful for RNNs like LSTMs.


7. Logging and Monitoring
The code uses logging utilities (logging and CSVLog) to track model configurations, training progress, and evaluation results.
System resources (e.g., GPU usage) are logged for reproducibility and analysis.


8. Main Function
The main() function orchestrates the training and testing processes, ensuring the model is trained on the dataset augmented with 
the pretrained embeddings and then evaluated on the test data. pretrained_mbeddings are loaded based on the specified type and 
integrated into the model's embedding layer. supervised_embeddings, if specified in the program arguments, are also integrated into the
neural model, adding semantic information from the labels directly into the embeddings. If specified in the arguments, the final 
embedding matrix can combine both pretrained and supervised embeddings, ensuring comprehensive input representations for the neural 
network. By combining pretrained and supervised embeddings, the model leverages rich, pre-learned semantic information from large 
corpora (via pretrained embeddings) and task-specific label information (via supervised embeddings) to enhance its performance on 
specific tasks.
"""


# ---------------------------------------------------------------------------------------------------------------------------

def init_Net(opt, nC, vocabsize, pretrained_embeddings, sup_range):
    
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

    print("Neural Classifier Model:\n", model)

    model.xavier_uniform()
    model = model.to(opt.device)

    if opt.tunable:
        print("Fine-tuning embeddings...")
        model.finetune_pretrained()

    print("emb_size:", model.get_embedding_size())
    print("learnable_emb_size:", model.get_learnable_embedding_size())

    embsizeX, embsizeY = model.get_embedding_size()
    print("embsizeX:", embsizeX)
    print("embsizeY:", embsizeY)

    lrnsizeX, lrnsizeY = model.get_learnable_embedding_size()
    print("lrnsizeX:", lrnsizeX)
    print("lrnsizeY:", lrnsizeY)

    return model, embsizeX, embsizeY, lrnsizeX, lrnsizeY

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


def init_loss(classification_type, device, class_weights):
    assert classification_type in ['multilabel','singlelabel'], 'unknown classification mode'

    
    # In your criterion (loss function) initialization, pass the class weights
    if classification_type == 'multilabel':
        L = torch.nn.BCEWithLogitsLoss()
    else:
        L = torch.nn.CrossEntropyLoss()
    
    """
    # In your criterion (loss function) initialization, pass the class weights
    if classification_type == 'multilabel':
        L = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights, device=opt.device))
    else:
        L = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=opt.device))
    """
    
    return L.to(device)



# ------------------------------------------------------------------------------------------------------------------------------------------------

def train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, dataset, epoch, method_name, loss_history):
    
    print("\t... training ...")

    #print("--model--\n", model)
    #print("train_index:", type(train_index))
    #print("ytr:", ytr)
    #print("pad_index:", type(pad_index))
    #print("method_name:", method_name)
    
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
                ytr = np.vstack((ytr[from_:], ytr[:to_]))
            else:
                ytr = np.concatenate((ytr[from_:], ytr[:to_]))
        else:
            train_index = train_index[from_:to_]
            ytr = ytr[from_:to_]

    model.train()

    # Initialize a variable to store # embedding dimensions
    """
    dims = model.embed.dim()
    print("dims:", {dims})
    """

    for idx, (batch, target) in enumerate(batchify(train_index, ytr, opt.batch_size, pad_index, opt.device, as_long)):

        """
        if dims is None:
            dims = batch.shape[1]  # Get the number of features (columns) from the batch
            print("# dimensions:", dims)
        """
        
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

    #logfile.insert(dimensions=dims, epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time.time() - tinit)

    return mean_loss

# ------------------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------------------------

def test(model, test_index, yte, pad_index, classification_type, tinit, epoch, logfile, criterion, measure_prefix, loss_history, embedding_size):
    
    print("\t..testing...")

    model.eval()
    predictions = []

    test_loss = 0
    total_batches = 0

    target_long = isinstance(criterion, torch.nn.CrossEntropyLoss)

    """
    # Initialize a variable to store # embedding dimensions
    dims = model.embed.dim()
    print("dims:", {dims})

    # Retrieve the number of dimensions from the first example in test_index
    if isinstance(test_index, list) and len(test_index) > 0:
        dims = len(test_index[0])  # Assuming each item in test_index is a list/vector of features
        print(f"Number of dimensions in test data: {dims}")
    else:
        print("Unable to determine the number of dimensions from test_index")
    """

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

    if (measure_prefix == 'final-te'):
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

        logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)
    

    mean_loss = test_loss / total_batches
    loss_history['test_loss'].append(mean_loss)

    if (measure_prefix == 'final-te'):
        logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-loss', value=mean_loss, timelapse=time.time() - tinit)

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
    

# --------------------------------------------------------------------------------------------------------------------------------------
#
# command line argument, program: parser plus assertions + main(opt)
#
# --------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    available_datasets = LCDataset.dataset_available
    available_dropouts = {'sup','none','full','learn'}

    
    print("\n\t\t------------------------------------- LAYER CAKE: Neural text classification with Word-Class Embeddings -------------------------------------\n")

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
    
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding - if larger, then PCA with this '
                             'number of components is applied (default 300)')
    
    parser.add_argument('--max-epoch-length', type=int, default=None, metavar='int',
                        help='number of (batched) training steps before considering an epoch over (None: full epoch)') #300 for wipo-sl-sc
    
    parser.add_argument('--force', action='store_true', default=False,
                        help='do not check if this experiment has already been run')
    
    parser.add_argument('--tunable', action='store_true', default=False,
                        help='pretrained embeddings are tunable from the beginning (default False, i.e., static)')
    
    parser.add_argument('--nozscore', action='store_true', default=False,
                        help='disables z-scoring form the computation of WCE')
    
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
    print("embeddings:", embeddings)    
    print("embedding_path:", emb_path)

    #
    # Load the dataset and the associated (pretrained) embedding structures
    # to be fed into the model
    #                                                          
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

    if (opt.pretrained is None):
        pretrained_vectors = None
        
    #word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(lcd, pretrained_vectors)

    if opt.pretrained in ['bert', 'roberta', 'xlnet', 'gpt2', 'llama']:
        toke = lcd.tokenizer
    else:
        toke = None

    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset=lcd, tokenizer=toke, max_length=lcd.max_length, pretrained=pretrained_vectors)
    
    print("word2index:", type(word2index), len(word2index))
    print("out_of_vocabulary:", type(out_of_vocabulary), len(out_of_vocabulary))

    print("training and validation data split...")

    #print("lcd.devel_target:", type(lcd.devel_target), lcd.devel_target.shape)

    val_size = min(int(len(devel_index) * .2), 20000)                   # dataset split tr/val/test

    # Perform the train_test_split with stratification based on the class labels

    train_index, val_index, ytr, yval = train_test_split(
        devel_index, lcd.devel_target, test_size=val_size, random_state=opt.seed, shuffle=True
    )

    print("lcd.devel_target:", type(lcd.devel_target), lcd.devel_target.shape)
    print("lcd.devel_target[0]:\n", type(lcd.devel_target[0]), lcd.devel_target[0])

    print("lcd.devel_labelmatrix:", type(lcd.devel_labelmatrix), lcd.devel_labelmatrix.shape)
    print("lcd.devel_labelmatrix[0]:\n", type(lcd.devel_labelmatrix[0]), lcd.devel_labelmatrix[0])

    # Compute class weights based on the training set
    # Convert sparse matrix to dense array
    dense_labels = lcd.devel_labelmatrix.toarray()

    class_weights = []
    for i in range(dense_labels.shape[1]):
        class_weight = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=dense_labels[:, i])
        class_weights.append(class_weight)

    # Convert to a tensor for PyTorch
    class_weights = torch.tensor(class_weights, device=opt.device)
#    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(lcd.devel_labelmatrix), y=lcd.devel_labelmatrix)
    #print("class_weights:", class_weights)

    yte = lcd.test_target

    """
    print("ytr:", type(ytr), ytr.shape)
    print("ytr:\n", ytr)

    print("yte:", type(yte), yte.shape)
    print("yte:\n", yte)
    """

    vocabsize = len(word2index) + len(out_of_vocabulary)
    print("vocabsize:", {vocabsize})

    pretrained_embeddings, sup_range = embedding_matrix(lcd, pretrained_vectors, vocabsize, word2index, out_of_vocabulary, opt)
    if pretrained_embeddings is not None:
        print("pretrained_embeddings:", type(pretrained_embeddings), pretrained_embeddings.shape)
    else:
        print("pretrained_embeddings: None")
    
    model, embedding_sizeX, embedding_sizeY, lrn_sizeX, lrn_sizeY = init_Net(opt, lcd.nC, vocabsize, pretrained_embeddings, sup_range)
    optim = init_optimizer(model, lr=opt.lr, weight_decay=opt.weight_decay)

    criterion = init_loss(lcd.classification_type, opt.device, class_weights)

    emb_size_str = f'({embedding_sizeX}, {embedding_sizeY})'
    print("emb_size:", emb_size_str)

    lrn_size_str = f'({lrn_sizeX}, {lrn_sizeY})'
    print("lrn_size:", lrn_size_str)

    emb_size_str = f'{emb_size_str}:{lrn_size_str}'

    embedding_dim = model.embed.dim()
    print("embedding_dim:", {embedding_dim})

    # train-validate
    tinit = time.time()
    create_if_not_exist(opt.checkpoint_dir)
    early_stop = EarlyStopping(model, patience=opt.patience, checkpoint=f'{opt.checkpoint_dir}/{opt.net}-{opt.dataset}')

    loss_history = {'train_loss': [], 'test_loss': []}              # Initialize loss tracking

    for epoch in range(1, opt.nepochs + 1):

        print(" \n-------------- EPOCH ", {epoch}, "-------------- ")    
        train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, opt.dataset, epoch, method_name, loss_history)
        
        macrof1, test_loss = test(model, val_index, yval, pad_index, lcd.classification_type, tinit, epoch, logfile, criterion, 'va', loss_history, embedding_size=emb_size_str)

        early_stop(macrof1, epoch)

        if opt.test_each>0:
            if (opt.plotmode and (epoch==1 or epoch%opt.test_each==0)) or (not opt.plotmode and epoch%opt.test_each==0 and epoch<opt.nepochs):
                test(model, test_index, yte, pad_index, lcd.classification_type, tinit, epoch, logfile, criterion, 'te', loss_history, embedding_size=emb_size_str)

        if early_stop.STOP:
            print('[early-stop]')
            if not opt.plotmode:                # with plotmode activated, early-stop is ignored
                break

    print("\t...restoring best model...")

    # restores the best model according to the Mf1 of the validation set (only when plotmode==False)
    stoptime = early_stop.stop_time - tinit
    stopepoch = early_stop.best_epoch

    logfile.insert(dimensions=emb_size_str, epoch=stopepoch, measure=f'early-stop', value=early_stop.best_score, timelapse=stoptime)

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
        test_loss = test(model, test_index, yte, pad_index, lcd.classification_type, tinit, epoch, logfile, criterion, 'final-te', loss_history, embedding_size=emb_size_str)


    if (opt.plotmode):                                          # Plot the training and testing loss after all epochs
        plot_loss_over_epochs( opt.dataset, {
            'epochs': np.arange(1, len(loss_history['train_loss']) + 1),
            'train_loss': loss_history['train_loss'],
            'test_loss': loss_history['test_loss']
        }, method_name, '../output')


    # --------------------------------------------------------------------------------------------------------------------------------------
