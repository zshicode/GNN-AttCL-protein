import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.ngram = 3
        self.beta = nn.Parameter(torch.FloatTensor(1))
        nn.init.normal_(self.beta,1,0.02)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.ngram,1)),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.BiLstm = nn.LSTM(
            input_size = dim,    # input size
            hidden_size = dim,   # hidden size of lstm
            num_layers = 1,     # num of layers
            bidirectional = True  # two directions of LSTM
        )
        #  output_size=hidden_size*num_direction
        self.lstmout = nn.Linear(2*dim, dim)
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        """Graph Isomorphism Network"""

        '''
        AI = A + torch.eye(A.shape[0]).to(device)
        D = torch.diag(torch.pow(AI.sum(dim=0),-0.5))
        A = D.mm(AI).mm(D)
        '''
        A = A + self.beta*torch.eye(A.shape[0]).to(device)
        for i in range(layer):
            hs = torch.matmul(A, xs)
            xs = torch.relu(self.W_gnn[i](hs))
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        xs = self.cnn(xs)
        xs = torch.mean(xs,2) # mean pooling
        xs,_ = self.BiLstm(xs)
        xs = self.lstmout(F.dropout(xs,0.2))
        xs = torch.squeeze(xs, 0)
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs
        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def fingerprint_one_hot(self,fingerprints):
        n = len(fingerprints)
        feat = torch.zeros(n,100).to(device)
        for i in range(n):
            feat[i,fingerprints[i]] = 1
        
        return feat


    def net(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        # for j in range(layer_output):
        #     cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        return interaction

    def forward(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.net(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
        AUC = roc_auc_score(T, S)
        P,R,TH = precision_recall_curve(T, S)
        AUPR = auc(R, P)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, AUPR, precision, recall

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    '''
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])
    '''
    DATASET='human'
    # DATASET=celegans
    # DATASET=yourdata

    # radius=1
    radius='2'
    # radius=3

    # ngram=2
    ngram='3'

    dim=16
    layer_gnn=2
    side=1
    window=2*side+1
    layer_cnn=2
    layer_output=2
    lr=1e-3
    lr_decay=0.5
    decay_interval=10
    weight_decay=1e-6
    iteration=100

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + radius + '_ngram' + ngram + '/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    #n_fingerprint = len(fingerprint_dict)
    n_fingerprint = 100
    #n_word = len(word_dict)
    n_word = 30
    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_test = split_dataset(dataset, 0.9)

    """Set a model."""
    torch.manual_seed(1234)
    model = CompoundProteinInteractionPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    
    """Output files."""
    # file_AUCs = '../output/result/AUCs--' + '.txt'
    # file_model = '../output/model/'
    # AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
    #         'AUC_test\tPrecision_test\tRecall_test')
    # with open(file_AUCs, 'w') as f:
    #     f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    #print(AUCs)
    start = timeit.default_timer()
    n_samples = len(interactions)
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)/n_samples
        AUC_test, AUPR_test, precision_test, recall_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start
        print('Epoch\tTime(sec)\tLoss_train\t'
             'AUC_test\tAUPR_test\tPrecision_test\tRecall_test')
        print([epoch, time, loss_train, 
                AUC_test, AUPR_test, precision_test, recall_test])
        # tester.save_AUCs(AUCs, file_AUCs)
        # tester.save_model(model, file_model)

        #print('\t'.join(map(str, AUCs)))
