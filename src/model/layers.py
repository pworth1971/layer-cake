"""

The EmbeddingCustom class is responsible for handling both pretrained and learnable embeddings, and it provides 
functionality for embedding dropout. Here’s a detailed explanation of its components:

Initialization (__init__ method):
- Pretrained Embeddings: If pretrained embeddings are provided, they are stored in a non-trainable nn.Embedding layer.
- Learnable Embeddings: If a learnable_length is specified, an additional nn.Embedding layer is created for learnable embeddings.
- Dropout Parameters: Handles the setup for embedding dropout based on the specified range and dropout probability.
- Embedding Length: The total embedding length is the sum of the pretrained and learnable embedding lengths.


Forward Pass (forward method):
- Embedding: The input is passed through the _embed method to get the combined embeddings.
- Dropout: The embeddings are passed through the _embedding_dropout method if dropout is specified.

Embedding Dropout (_embedding_dropout and _embedding_dropout_range methods):
- Full Dropout: If drop_embedding_range is None, dropout is applied to the entire embedding.
- Range Dropout: If a range is specified, dropout is applied only to the specified range of the embedding dimensions.

Embedding Combination (_embed method):
Combines the pretrained and learnable embeddings based on the provided input indices.



--------------------------------------------------------------------------------------------------------------------------------
Pretrained Embeddings Integration
The NeuralClassifier class uses a custom embedding layer, EmbeddingCustom, to incorporate pretrained embeddings. 

Embedding Layer Initialization:
The EmbeddingCustom class is initialized with the pretrained embeddings, if provided. The embeddings are passed 
during the instantiation of NeuralClassifier:

self.embed = EmbeddingCustom(vocab_size, learnable_length, pretrained, drop_embedding_range, drop_embedding_prop)


Forward Pass:
In the forward method, the input is passed through the embedding layer to get the word embeddings:

def forward(self, input):
    word_emb = self.embed(input)
    doc_emb = self.projection(word_emb)
    logits = self.label(doc_emb)
    return logits


Finetuning:
If the tunable flag is set, the finetune_pretrained method is called to enable fine-tuning of the pretrained embeddings:
def finetune_pretrained(self):
    self.embed.finetune_pretrained()


The EmbeddingCustom layer is initialized with the following parameters:
- vocab_size: Size of the vocabulary.
- learnable_length: Length of the learnable embeddings.
- pretrained: The pretrained embeddings.
- drop_embedding_range: The range for dropout in the embedding layer.
- drop_embedding_prop: Dropout probability for the embedding layer.


Forward Method:
During the forward pass, the input is converted into word embeddings using the EmbeddingCustom layer.

--------------------------------------------------------------------------------------------------------------------------------
Here is the overall process:

Loading Pretrained Embeddings:
Pretrained embeddings are loaded and combined (if supervised embeddings are also used) in the embedding_matrix 
function. This combined embedding matrix is returned to be used for model initialization.

Initializing NeuralClassifier:
The NeuralClassifier is initialized with the combined embeddings passed to the EmbeddingCustom layer. This 
integration ensures that the pretrained embeddings are used to represent the input text data.

Training and Forward Pass:
During training and evaluation, the input data is passed through the EmbeddingCustom layer to get word embeddings, which 
are then processed by the projection and label layers to produce logits.
--------------------------------------------------------------------------------------------------------------------------------
"""


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class EmbeddingCustom(nn.Module):

    def __init__(self,
                 vocab_size,
                 learnable_length,
                 pretrained=None,
                 drop_embedding_range=None,
                 drop_embedding_prop=0):

        super(EmbeddingCustom, self).__init__()
        assert 0 <= drop_embedding_prop <= 1, 'drop_embedding_prop: wrong range'

        print("EmbeddingCustom: __init__()")

        self.vocab_size = vocab_size
        self.drop_embedding_range = drop_embedding_range
        self.drop_embedding_prop = drop_embedding_prop

        self.ptX = 0
        self.ptY = 0
        self.lrnX = 0
        self.lrnY = 0

        pretrained_embeddings = None
        pretrained_length = 0
        if pretrained is not None:
            pretrained_length = pretrained.shape[1]
            assert pretrained.shape[0] == vocab_size, \
                f'pre-trained matrix (shape {pretrained.shape}) does not match with the vocabulary size {vocab_size}'
            pretrained_embeddings = nn.Embedding(vocab_size, pretrained_length)
            self.ptX = vocab_size
            self.ptY = pretrained_length
            # by default, pretrained embeddings are static; this can be modified by calling finetune_pretrained()
            pretrained_embeddings.weight = nn.Parameter(pretrained, requires_grad=False)

        learnable_embeddings = None
        if learnable_length > 0:
            learnable_embeddings = nn.Embedding(vocab_size, learnable_length)
            self.lrnX = vocab_size
            self.lrnY = learnable_length

        embedding_length = learnable_length + pretrained_length
        assert embedding_length > 0, '0-size embeddings'

        self.pretrained_embeddings = pretrained_embeddings
        self.learnable_embeddings = learnable_embeddings
        self.embedding_length = embedding_length

        if (self.pretrained_embeddings is None):
            print("pretrained_embeddings: None")
        else:    
            print("pretrained_embeddings:", self.pretrained_embeddings)
        
        if (self.learnable_embeddings is None):
            print("learnable_embeddings: None")
        else:
            print("learnable_embeddings:", self.learnable_embeddings)
    
        print("embedding_length:", self.embedding_length)

        assert self.drop_embedding_range is None or \
               (0<=self.drop_embedding_range[0]<self.drop_embedding_range[1]<=embedding_length), \
            'dropout limits out of range'

    def get_pt_dimensions(self):
        return self.ptX, self.ptY

    def get_lrn_dimensions(self):
        return self.lrnX, self.lrnY

    def forward(self, input):
        input = self._embed(input)
        input = self._embedding_dropout(input)
        return input

    def finetune_pretrained(self):
        self.pretrained_embeddings.requires_grad = True
        self.pretrained_embeddings.weight.requires_grad = True

    def _embedding_dropout(self, input):
        if self.droprange():
            return self._embedding_dropout_range(input)
        elif self.dropfull():
            return F.dropout(input, p=self.drop_embedding_prop, training=self.training)
        return input

    def _embedding_dropout_range(self, input):
        drop_range = self.drop_embedding_range
        p = self.drop_embedding_prop
        if p > 0 and self.training and drop_range is not None:
            drop_from, drop_to = drop_range
            m = drop_to - drop_from     #length of the supervised embedding (or the range)
            l = input.shape[2]          #total embedding length
            corr = (1 - p)
            input[:, :, drop_from:drop_to] = corr * F.dropout(input[:, :, drop_from:drop_to], p=p)
            input /= (1 - (p * m / l))

        return input

    def _embed(self, input):
        input_list = []
        if self.pretrained_embeddings:
            input_list.append(self.pretrained_embeddings(input))
        if self.learnable_embeddings:
            input_list.append(self.learnable_embeddings(input))
        return torch.cat(tensors=input_list, dim=2)

    def dim(self):
        return self.embedding_length

    def dropnone(self):
        if self.drop_embedding_prop == 0:
            return True
        if self.drop_embedding_range is None:
            return True
        return False

    def dropfull(self):
        if self.drop_embedding_prop == 0:
            return False
        if self.drop_embedding_range == [0, self.dim()]:
            return True
        return False

    def droprange(self):
        if self.drop_embedding_prop == 0:
            return False
        if self.drop_embedding_range is None:
            return False
        if self.dropfull():
            return False
        return True


class CNNprojection(nn.Module):

    def __init__(self, embedding_dim, out_channels, kernel_heights=[3, 5, 7], stride=1, padding=0, drop_prob=0.5):
        super(CNNprojection, self).__init__()
        in_channels = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_dim), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_dim), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_dim), stride, padding)
        self.dropout = nn.Dropout(drop_prob)
        self.out_dimensions = len(kernel_heights) * out_channels

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, input): # input.size() = (batch_size, num_seq, embedding_dim)
        input = input.unsqueeze(1)  # input.size() = (batch_size, 1, num_seq, embedding_length)

        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)  # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)  # fc_in.size()) = (batch_size, num_kernels*out_channels)
        return fc_in

    def dim(self):
        return self.out_dimensions


class LSTMprojection(nn.Module):

    def __init__(self, embedding_dim, hidden_size):
        super(LSTMprojection, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_dim, hidden_size)

    def forward(self, input):  # input.size() = (batch_size, num_seq, embedding_dim)
        batch_size = input.shape[0]
        input = input.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        return final_hidden_state[-1]

    def dim(self):
        return self.hidden_size



# Setup device prioritizing CUDA, then MPS, then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
class ATTNprojection(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_size):
        super(ATTNprojection, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_dim, hidden_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input):  # input.size() = (batch_size, num_seq, embedding_dim)
        batch_size = input.shape[0]
        input = input.permute(1, 0, 2)
        """
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        """
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
        
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)
        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output

    def dim(self):
        return self.hidden_size

