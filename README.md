# Graph neural networks and attention-based CNN-LSTM for protein classification

This paper proposes three models for protein classification. Firstly, this paper proposes a Multi-label CAZyme classification model using CNN-LSTM with Attention mechanism.  Secondly, this paper proposes a variational graph autoencoder based subspace learning model for protein graph classification. Thirdly, this paper proposes  Graph Isomorphism Networks and Attention-based CNN-LSTM for compound-protein interactions prediction. 

## Usage

### CLACAZy

We select 4683 CAZymes from all 1,066,327 CAZymes, to construct the dataset. 3759 CAZymes are divided into training set, while the other 924 CAZymes are divided into testing set. See `train.fasta` and `test.fasta` .

The code can be run through
```
cd CLACAZy
python main.py
```

Hyper parameter setting can refer to our code repository.

### VSPool

Data formed for VSPool experiments can be downloaded from this link: (https://github.com/HongyangGao/Graph-U-Nets/tree/master/data).

After downloading the data, the code can be run through
```
cd VSPool
```
Type
```
./run_GNN.sh DATA FOLD GPU
```
to run on dataset using fold number (1-10).

You can run
```
./run_GNN.sh DD 0 0
```
to run on DD dataset with 10-fold cross validation on GPU #0.

### GACLCPI

The code can be run through
```
cd GACLCPI/code
python preprocess_data.py
python run_training.py
```

### Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy
pytorch
sklearn
biopython
biovec
rdkit
```
